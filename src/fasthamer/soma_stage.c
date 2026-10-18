// soma_stage.c -- SOMA-X right-hand stage (identity -> skeleton fit -> repose ->
// FK + LBS) in plain C, a port of fasthamer/soma_np.py (itself a port of
// py-soma-x's SOMAHandLayer.prepare_identity() + pose()). Double precision,
// single-threaded; a batch of B hands runs in ~0.1 ms.
//
// Build: cc -O3 -shared -fPIC soma_stage.c -o soma_stage.so (done at runtime by
// fasthamer.soma_np, cached in ~/.cache/fasthamer).
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int V, J, F, K;
    const double *pca_mean, *pca_matrix; double rest_scale;
    const double *Wpos, *bpos;
    const int *parents, *fit_joints, *set_off, *set_vids;
    const double *p0, *R0, *bind_shape;
    const int *child_off, *child_ids;
    const int *skin_off, *skin_j; const double *skin_w;
    const double *bind_local_R, *bind_root_t;
    double prior_strength, rank_threshold; int ns_iters, skip_endjoints;
    // scratch
    double *rest, *rest_bind, *t, *R, *Rw, *tw, *Rb, *tb, *lt, *bone_R, *bone_t;
} soma_ctx;

// ---------------------------------------------------------------- mat3 helpers
static void m3_mul(const double *A, const double *B, double *C) {           // C = A B
    double T[9];
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++)
        T[3*i+j] = A[3*i]*B[j] + A[3*i+1]*B[3+j] + A[3*i+2]*B[6+j];
    memcpy(C, T, sizeof T);
}
static void m3_mulT(const double *A, const double *B, double *C) {          // C = A^T B
    double T[9];
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++)
        T[3*i+j] = A[i]*B[j] + A[3+i]*B[3+j] + A[6+i]*B[6+j];
    memcpy(C, T, sizeof T);
}
static void m3_mulBT(const double *A, const double *B, double *C) {         // C = A B^T
    double T[9];
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++)
        T[3*i+j] = A[3*i]*B[3*j] + A[3*i+1]*B[3*j+1] + A[3*i+2]*B[3*j+2];
    memcpy(C, T, sizeof T);
}
static void m3_vec(const double *A, const double *x, double *y) {           // y = A x
    double t0 = A[0]*x[0]+A[1]*x[1]+A[2]*x[2], t1 = A[3]*x[0]+A[4]*x[1]+A[5]*x[2], t2 = A[6]*x[0]+A[7]*x[1]+A[8]*x[2];
    y[0] = t0; y[1] = t1; y[2] = t2;
}
static void m3_Tvec(const double *A, const double *x, double *y) {          // y = A^T x
    double t0 = A[0]*x[0]+A[3]*x[1]+A[6]*x[2], t1 = A[1]*x[0]+A[4]*x[1]+A[7]*x[2], t2 = A[2]*x[0]+A[5]*x[1]+A[8]*x[2];
    y[0] = t0; y[1] = t1; y[2] = t2;
}
static double m3_det(const double *M) {
    return M[0]*(M[4]*M[8]-M[5]*M[7]) - M[1]*(M[3]*M[8]-M[5]*M[6]) + M[2]*(M[3]*M[7]-M[4]*M[6]);
}
static void v3_cross(const double *a, const double *b, double *c) {
    double x = a[1]*b[2]-a[2]*b[1], y = a[2]*b[0]-a[0]*b[2], z = a[0]*b[1]-a[1]*b[0];
    c[0] = x; c[1] = y; c[2] = z;
}
static double v3_norm(const double *a) { return sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]); }

// ------------------------------------------------- rotation fitting (SOMA-X 'auto')
// Shortest-arc rotation R with R b ~= a (soma.geometry.transforms.rodrigues_rotation).
static void rodrigues(const double *a, const double *b, double *R, double eps) {
    double au[3], bu[3], na = v3_norm(a), nb = v3_norm(b);
    na = na < eps ? eps : na; nb = nb < eps ? eps : nb;
    for (int i = 0; i < 3; i++) { au[i] = a[i]/na; bu[i] = b[i]/nb; }
    double dot = au[0]*bu[0]+au[1]*bu[1]+au[2]*bu[2];
    if (dot > 1.0) dot = 1.0; if (dot < -1.0) dot = -1.0;
    double v[3]; v3_cross(bu, au, v);
    if (dot < -1.0 + 1e-6) {                       // antiparallel: 180 deg about a perpendicular axis
        double w[3] = {1, 0, 0}; if (fabs(bu[0]) > 0.6) { w[0] = 0; w[1] = 1; }
        double ax[3]; v3_cross(bu, w, ax); double n = v3_norm(ax);
        for (int i = 0; i < 3; i++) ax[i] /= n;
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) R[3*i+j] = 2.0*ax[i]*ax[j] - (i == j);
        return;
    }
    double K[9] = {0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0}, KK[9];
    m3_mul(K, K, KK);
    double f = 1.0 / (1.0 + dot);
    for (int i = 0; i < 9; i++) R[i] = (i % 4 == 0) + K[i] + f*KK[i];
}

// Jacobi eigen-decomposition of a symmetric 3x3: A = Q diag(w) Q^T (Q columns = eigenvectors).
static void sym3_eig(const double *A, double *w, double *Q) {
    double a[9]; memcpy(a, A, sizeof a);
    for (int i = 0; i < 9; i++) Q[i] = (i % 4 == 0);
    for (int sweep = 0; sweep < 50; sweep++) {
        double off = fabs(a[1]) + fabs(a[2]) + fabs(a[5]);
        if (off < 1e-30) break;
        for (int p = 0; p < 3; p++) for (int q = p + 1; q < 3; q++) {
            if (fabs(a[3*p+q]) < 1e-300) continue;
            double theta = (a[3*q+q] - a[3*p+p]) / (2.0 * a[3*p+q]);
            double tt = (theta >= 0 ? 1.0 : -1.0) / (fabs(theta) + sqrt(theta*theta + 1.0));
            double c = 1.0 / sqrt(tt*tt + 1.0), s = tt * c;
            for (int k = 0; k < 3; k++) {            // rotate rows/cols p,q
                double akp = a[3*k+p], akq = a[3*k+q];
                a[3*k+p] = c*akp - s*akq; a[3*k+q] = s*akp + c*akq;
            }
            for (int k = 0; k < 3; k++) {
                double apk = a[3*p+k], aqk = a[3*q+k];
                a[3*p+k] = c*apk - s*aqk; a[3*q+k] = s*apk + c*aqk;
            }
            for (int k = 0; k < 3; k++) {
                double qkp = Q[3*k+p], qkq = Q[3*k+q];
                Q[3*k+p] = c*qkp - s*qkq; Q[3*k+q] = s*qkp + c*qkq;
            }
        }
    }
    for (int i = 0; i < 3; i++) w[i] = a[4*i];
}

// Kabsch via eigen-decomposition of H^T H (fallback when Newton-Schulz is invalid).
static void kabsch(const double *H, double *R) {
    double A[9], w[3], Q[9]; m3_mulT(H, H, A); sym3_eig(A, w, Q);
    int idx[3] = {0, 1, 2};                        // sort eigenvalues descending
    for (int i = 0; i < 3; i++) for (int j = i + 1; j < 3; j++) if (w[idx[j]] > w[idx[i]]) { int t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
    double U[9], Vt[9], s[3];
    for (int c = 0; c < 3; c++) {
        int e = idx[c]; s[c] = sqrt(w[e] > 0 ? w[e] : 0);
        double q[3] = {Q[e], Q[3+e], Q[6+e]}, u[3]; m3_vec(H, q, u);
        for (int i = 0; i < 3; i++) { U[3*i+c] = s[c] > 1e-12 ? u[i]/s[c] : 0.0; Vt[3*c+i] = q[i]; }
    }
    if (s[2] <= 1e-12) {                           // complete U with the cross product
        double u0[3] = {U[0], U[3], U[6]}, u1[3] = {U[1], U[4], U[7]}, u2[3]; v3_cross(u0, u1, u2);
        for (int i = 0; i < 3; i++) U[3*i+2] = u2[i];
    }
    double UVt[9]; m3_mul(U, Vt, UVt);
    if (m3_det(UVt) < 0) for (int i = 0; i < 3; i++) U[3*i+2] = -U[3*i+2];
    m3_mul(U, Vt, R);
}

// align_vectors(method='auto') on a covariance H (regularisation + Newton-Schulz polar
// factor + validity check + Kabsch fallback), as in soma.geometry.transforms.
static void rotation_from_covariance(const soma_ctx *c, const double *Hin, double *R) {
    const double eps = 1e-8;
    double H[9]; memcpy(H, Hin, sizeof H);
    double ps = 0; for (int i = 0; i < 3; i++) { double r = fabs(H[3*i])+fabs(H[3*i+1])+fabs(H[3*i+2]); if (r > ps) ps = r; }
    if (ps < eps) ps = eps;
    double vol = fabs(m3_det(H)) / (ps*ps*ps);
    double rw = (c->rank_threshold - vol) / c->rank_threshold; rw = rw < 0 ? 0 : (rw > 1 ? 1 : rw);
    double add = c->prior_strength * rw * ps;
    H[0] += add; H[4] += add; H[8] += add;
    // Newton-Schulz
    double mx = 0; for (int i = 0; i < 3; i++) { double r = fabs(H[3*i])+fabs(H[3*i+1])+fabs(H[3*i+2]); if (r > mx) mx = r; }
    for (int i = 0; i < 9; i++) R[i] = H[i] / (mx + eps);
    for (int it = 0; it < c->ns_iters; it++) {
        double RtR[9], T[9]; m3_mulT(R, R, RtR);
        for (int i = 0; i < 9; i++) T[i] = 3.0*(i % 4 == 0) - RtR[i];
        m3_mul(R, T, R); for (int i = 0; i < 9; i++) R[i] *= 0.5;
    }
    if (m3_det(R) < 0) for (int i = 0; i < 3; i++) R[3*i+2] = -R[3*i+2];
    // validity (as rotation_matrices_are_valid) + det(H_auto) < 0 -> Kabsch
    double d = m3_det(R), RtR[9]; m3_mulT(R, R, RtR); double ortho = 0; int finite = 1;
    for (int i = 0; i < 9; i++) { if (!isfinite(R[i])) finite = 0; double e = fabs(RtR[i] - (i % 4 == 0)); if (e > ortho) ortho = e; }
    int valid = finite && isfinite(d) && d > 0 && fabs(d - 1.0) <= 1e-2 && ortho <= 1e-2;
    if (!valid || m3_det(H) < 0) kabsch(H, R);
}

// SOMA's virtual-normal conditioning term built from the first two correspondences
// (A, Bv: at least 2 x 3): H += (n_src |a0|/|n_src|) (n_dst |b0|/|n_dst|)^T
static void virtual_normal(const double *A, const double *Bv, double *H) {
    const double eps = 1e-8;
    double ns[3], nd[3]; v3_cross(A, A+3, ns); v3_cross(Bv, Bv+3, nd);
    double ls = v3_norm(ns), ld = v3_norm(nd);
    if (ls > 1e-9 && ld > 1e-9) {
        double ss = v3_norm(A) / (ls + eps), sd = v3_norm(Bv) / (ld + eps);
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) H[3*i+j] += ns[i]*ss * nd[j]*sd;
    }
}
// covariance H = sum_n a_n b_n^T (+ the virtual-normal term when n >= 2)
static void covariance(const double *A, const double *Bv, int n, double *H) {
    memset(H, 0, 9 * sizeof(double));
    for (int k = 0; k < n; k++) for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) H[3*i+j] += A[3*k+i]*Bv[3*k+j];
    if (n >= 2) virtual_normal(A, Bv, H);
}

// ------------------------------------------------------------------ kinematics
// world[j] = world[parent] * [localR[j] | localt[j]]; root = [localR[0] | localt[0]]
static void fk(const soma_ctx *c, const double *localR, const double *localt, double *Rw, double *tw) {
    memcpy(Rw, localR, 9 * sizeof(double)); memcpy(tw, localt, 3 * sizeof(double));
    for (int j = 1; j < c->J; j++) {
        int p = c->parents[j];
        m3_mul(Rw + 9*p, localR + 9*j, Rw + 9*j);
        double v[3]; m3_vec(Rw + 9*p, localt + 3*j, v);
        for (int i = 0; i < 3; i++) tw[3*j+i] = v[i] + tw[3*p+i];
    }
}
// local translations of world transforms: R_parent^T (t_j - t_parent); root keeps t_0
static void local_translations(const soma_ctx *c, const double *Rw, const double *tw, double *lt) {
    memcpy(lt, tw, 3 * sizeof(double));
    for (int j = 1; j < c->J; j++) {
        int p = c->parents[j]; double d[3] = {tw[3*j]-tw[3*p], tw[3*j+1]-tw[3*p+1], tw[3*j+2]-tw[3*p+2]};
        m3_Tvec(Rw + 9*p, d, lt + 3*j);
    }
}
// bone_j = T_world[j] * inv(T_bind[j]); verts = sum_j w_vj (R_j x_v + t_j)
static void lbs(const soma_ctx *c, const double *x, const double *Rw, const double *tw,
                const double *Rbind, const double *tbind, double *bR, double *bt, double *out) {
    for (int j = 0; j < c->J; j++) {
        m3_mulBT(Rw + 9*j, Rbind + 9*j, bR + 9*j);
        double v[3]; m3_vec(bR + 9*j, tbind + 3*j, v);
        for (int i = 0; i < 3; i++) bt[3*j+i] = tw[3*j+i] - v[i];
    }
    for (int v = 0; v < c->V; v++) {
        double acc[3] = {0, 0, 0};
        for (int k = c->skin_off[v]; k < c->skin_off[v+1]; k++) {
            int j = c->skin_j[k]; double w = c->skin_w[k], y[3]; m3_vec(bR + 9*j, x + 3*v, y);
            for (int i = 0; i < 3; i++) acc[i] += w * (y[i] + bt[3*j+i]);
        }
        for (int i = 0; i < 3; i++) out[3*v+i] = acc[i];
    }
}

// ----------------------------------------------------------------------- API
void *soma_create(int V, int J, int F, int K,
                  const double *pca_mean, const double *pca_matrix, double rest_scale,
                  const double *Wpos, const double *bpos, const int *parents,
                  const int *fit_joints, const int *set_off, const int *set_vids,
                  const double *p0, const double *R0, const double *bind_shape,
                  const int *child_off, const int *child_ids,
                  const int *skin_off, const int *skin_j, const double *skin_w,
                  const double *bind_local_R, const double *bind_root_t,
                  double prior_strength, double rank_threshold, int ns_iters, int skip_endjoints) {
    soma_ctx *c = (soma_ctx *)calloc(1, sizeof(soma_ctx));
    c->V = V; c->J = J; c->F = F; c->K = K;
    c->pca_mean = pca_mean; c->pca_matrix = pca_matrix; c->rest_scale = rest_scale;
    c->Wpos = Wpos; c->bpos = bpos; c->parents = parents; c->fit_joints = fit_joints;
    c->set_off = set_off; c->set_vids = set_vids; c->p0 = p0; c->R0 = R0; c->bind_shape = bind_shape;
    c->child_off = child_off; c->child_ids = child_ids; c->skin_off = skin_off; c->skin_j = skin_j; c->skin_w = skin_w;
    c->bind_local_R = bind_local_R; c->bind_root_t = bind_root_t;
    c->prior_strength = prior_strength; c->rank_threshold = rank_threshold; c->ns_iters = ns_iters; c->skip_endjoints = skip_endjoints;
    c->rest = (double *)malloc(sizeof(double) * V * 3); c->rest_bind = (double *)malloc(sizeof(double) * V * 3);
    c->t = (double *)malloc(sizeof(double) * J * 3); c->R = (double *)malloc(sizeof(double) * J * 9);
    c->Rw = (double *)malloc(sizeof(double) * J * 9); c->tw = (double *)malloc(sizeof(double) * J * 3);
    c->Rb = (double *)malloc(sizeof(double) * J * 9); c->tb = (double *)malloc(sizeof(double) * J * 3);
    c->lt = (double *)malloc(sizeof(double) * J * 3);
    c->bone_R = (double *)malloc(sizeof(double) * J * 9); c->bone_t = (double *)malloc(sizeof(double) * J * 3);
    return c;
}

void soma_destroy(void *ctx) {
    soma_ctx *c = (soma_ctx *)ctx; if (!c) return;
    free(c->rest); free(c->rest_bind); free(c->t); free(c->R); free(c->Rw); free(c->tw);
    free(c->Rb); free(c->tb); free(c->lt); free(c->bone_R); free(c->bone_t); free(c);
}

// pose (B,J,3,3) absolute-local rotations, root_t (B,3), identity (B,K), scales (B,J-1) or NULL
// -> verts (B,V,3), joints (B,J,3)
// Like soma_pose(), with optional per-vertex offsets (B,V,3) added to the bind-pose rest shape
// right before skinning: the learned mano2soma-x correctives (same injection point as SOMA-X's own).
void soma_pose_offsets(void *ctx, int B, const double *pose, const double *root_t, const double *identity,
                       const double *scales, const double *offsets, double *out_verts, double *out_joints) {
    soma_ctx *c = (soma_ctx *)ctx;
    const int V = c->V, J = c->J, K = c->K;
    double *rest = c->rest, *rest_bind = c->rest_bind, *t = c->t, *R = c->R;
    double *Rw = c->Rw, *tw = c->tw, *Rb = c->Rb, *tb = c->tb, *lt = c->lt;
    for (int b = 0; b < B; b++) {
        const double *id = identity + b*K;
        // 1. identity PCA rest shape
        for (int i = 0; i < V*3; i++) {
            double s = c->pca_mean[i];
            for (int k = 0; k < K; k++) s += id[k] * c->pca_matrix[(size_t)k*V*3 + i];
            rest[i] = s * c->rest_scale;
        }
        // 2. joint positions (linear regressor)
        for (int j = 0; j < J; j++) {
            double a0 = 0, a1 = 0, a2 = 0; const double *w = c->Wpos + (size_t)j*V;
            for (int v = 0; v < V; v++) { a0 += w[v]*rest[3*v]; a1 += w[v]*rest[3*v+1]; a2 += w[v]*rest[3*v+2]; }
            t[3*j] = a0 + c->bpos[3*j]; t[3*j+1] = a1 + c->bpos[3*j+1]; t[3*j+2] = a2 + c->bpos[3*j+2];
        }
        // 3. joint orientations, in joint index order (end joints copy the parent's fitted R)
        memcpy(R, c->R0, sizeof(double) * J * 9);
        int f = 0;
        for (int j = 0; j < J; j++) {
            int nch = c->child_off[j+1] - c->child_off[j];
            if (nch == 0 && c->skip_endjoints) { memcpy(R + 9*j, R + 9*c->parents[j], 9 * sizeof(double)); continue; }
            // R_init: Procrustes over the skinned vertex set
            double H[9] = {0}; int s0 = c->set_off[f], s1 = c->set_off[f+1];
            double A2[6], B2[6];
            for (int s = s0; s < s1; s++) {
                int v = c->set_vids[s];
                double a[3] = {rest[3*v]-t[3*j], rest[3*v+1]-t[3*j+1], rest[3*v+2]-t[3*j+2]};
                double bb[3] = {c->bind_shape[3*v]-c->p0[3*j], c->bind_shape[3*v+1]-c->p0[3*j+1], c->bind_shape[3*v+2]-c->p0[3*j+2]};
                for (int i = 0; i < 3; i++) for (int k = 0; k < 3; k++) H[3*i+k] += a[i]*bb[k];
                if (s - s0 < 2) { memcpy(A2 + 3*(s-s0), a, sizeof a); memcpy(B2 + 3*(s-s0), bb, sizeof bb); }
            }
            if (s1 - s0 >= 2) virtual_normal(A2, B2, H);   // SOMA's conditioning term from the first two pairs
            double Rinit[9]; rotation_from_covariance(c, H, Rinit);
            // children alignment
            double Al[9];
            if (nch == 1) {
                int ch = c->child_ids[c->child_off[j]];
                double po0[3] = {c->p0[3*ch]-c->p0[3*j], c->p0[3*ch+1]-c->p0[3*j+1], c->p0[3*ch+2]-c->p0[3*j+2]}, po[3];
                m3_vec(Rinit, po0, po);
                double pn[3] = {t[3*ch]-t[3*j], t[3*ch+1]-t[3*j+1], t[3*ch+2]-t[3*j+2]};
                rodrigues(pn, po, Al, 1e-8);
            } else {
                double An[3*16], Bn[3*16]; int n = nch > 16 ? 16 : nch;
                for (int q = 0; q < n; q++) {
                    int ch = c->child_ids[c->child_off[j] + q];
                    double po0[3] = {c->p0[3*ch]-c->p0[3*j], c->p0[3*ch+1]-c->p0[3*j+1], c->p0[3*ch+2]-c->p0[3*j+2]};
                    m3_vec(Rinit, po0, Bn + 3*q);
                    for (int i = 0; i < 3; i++) An[3*q+i] = t[3*ch+i] - t[3*j+i];
                }
                double Hc[9]; covariance(An, Bn, n, Hc); rotation_from_covariance(c, Hc, Al);
            }
            double T1[9]; m3_mul(Al, Rinit, T1); m3_mul(T1, c->R0 + 9*j, R + 9*j);
            f++;
        }
        // 4. fitted bind transforms T_fit = [R | t] -> local translations
        local_translations(c, R, t, lt);
        // 5. repose to the canonical bind pose
        memcpy(lt, c->bind_root_t, 3 * sizeof(double));
        fk(c, c->bind_local_R, lt, Rw, tw);                       // T_bind
        lbs(c, rest, Rw, tw, R, t, c->bone_R, c->bone_t, rest_bind);
        if (offsets) { const double *o = offsets + (size_t)b*V*3; for (int i = 0; i < V*3; i++) rest_bind[i] += o[i]; }
        memcpy(Rb, Rw, sizeof(double) * J * 9); memcpy(tb, tw, sizeof(double) * J * 3);
        // 6. pose: bone-scaled local translations of T_bind, root = root_t
        local_translations(c, Rb, tb, lt);
        if (scales) for (int j = 1; j < J; j++) { double s = scales[b*(J-1) + j-1]; lt[3*j] *= s; lt[3*j+1] *= s; lt[3*j+2] *= s; }
        memcpy(lt, root_t + 3*b, 3 * sizeof(double));
        fk(c, pose + (size_t)b*J*9, lt, Rw, tw);                   // T_world
        lbs(c, rest_bind, Rw, tw, Rb, tb, c->bone_R, c->bone_t, out_verts + (size_t)b*V*3);
        memcpy(out_joints + (size_t)b*J*3, tw, sizeof(double) * J * 3);
    }
}

void soma_pose(void *ctx, int B, const double *pose, const double *root_t, const double *identity,
               const double *scales, double *out_verts, double *out_joints) {
    soma_pose_offsets(ctx, B, pose, root_t, identity, scales, NULL, out_verts, out_joints);
}
