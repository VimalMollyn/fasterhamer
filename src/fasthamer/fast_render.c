// fast_render.c — MANO-mesh rasterizer for realtime HaMeR overlay, with
// supersampled anti-aliasing (SSAA) so silhouettes are smooth (no "crunchy"
// edges). Pinhole projection (matches hamer keypoint projection: u=fx*x/z+cx),
// per-vertex smooth (gouraud) shading via a camera headlight with a soft floor,
// z-buffer, coverage-based alpha composite onto an RGB uint8 image in place.
//
// Build: cc -O3 -ffast-math -shared -fPIC fast_render.c -o fast_render_c.so
#include <math.h>
#include <stdlib.h>

static inline float fminf3(float a, float b, float c) { return fminf(a, fminf(b, c)); }
static inline float fmaxf3(float a, float b, float c) { return fmaxf(a, fmaxf(b, c)); }

void rasterize_mesh(
    const float* verts,   // N*3, camera space (already + cam_t)
    int N,
    const int* faces,     // F*3 (int32)
    int F,
    float fx, float fy, float cx, float cy,
    int W, int H,
    unsigned char* img,   // H*W*3 RGB, modified in place
    float cr, float cg, float cb,   // mesh base color 0..1
    float ambient, float alpha,
    int ss)               // supersample factor (1 = off, 2/3 = AA)
{
    if (ss < 1) ss = 1;
    const int SW = W * ss, SH = H * ss;
    const float sfx = fx * ss, sfy = fy * ss, scx = cx * ss, scy = cy * ss;

    float* z  = (float*)malloc(sizeof(float) * N);
    float* px = (float*)malloc(sizeof(float) * N);
    float* py = (float*)malloc(sizeof(float) * N);
    float* nx = (float*)calloc(N, sizeof(float));
    float* ny = (float*)calloc(N, sizeof(float));
    float* nz = (float*)calloc(N, sizeof(float));
    float* zbuf = (float*)malloc(sizeof(float) * SW * SH);
    // supersampled mesh layer: shaded color (premultiplied not needed) + coverage
    float* sc = (float*)malloc(sizeof(float) * SW * SH * 3);
    unsigned char* cov = (unsigned char*)calloc(SW * SH, 1);
    for (int i = 0; i < SW * SH; i++) zbuf[i] = 1e30f;

    for (int i = 0; i < N; i++) {
        float X = verts[3*i], Y = verts[3*i+1], Z = verts[3*i+2];
        if (Z < 1e-4f) Z = 1e-4f;
        z[i] = Z; px[i] = sfx * X / Z + scx; py[i] = sfy * Y / Z + scy;
    }
    for (int f = 0; f < F; f++) {
        int a = faces[3*f], b = faces[3*f+1], c = faces[3*f+2];
        float e1x = verts[3*b]-verts[3*a], e1y = verts[3*b+1]-verts[3*a+1], e1z = verts[3*b+2]-verts[3*a+2];
        float e2x = verts[3*c]-verts[3*a], e2y = verts[3*c+1]-verts[3*a+1], e2z = verts[3*c+2]-verts[3*a+2];
        float fnx = e1y*e2z - e1z*e2y, fny = e1z*e2x - e1x*e2z, fnz = e1x*e2y - e1y*e2x;
        nx[a]+=fnx; ny[a]+=fny; nz[a]+=fnz;
        nx[b]+=fnx; ny[b]+=fny; nz[b]+=fnz;
        nx[c]+=fnx; ny[c]+=fny; nz[c]+=fnz;
    }
    for (int i = 0; i < N; i++) {
        float L = sqrtf(nx[i]*nx[i] + ny[i]*ny[i] + nz[i]*nz[i]) + 1e-9f;
        nz[i] /= L;  // z-component of the unit normal (headlight term)
    }

    float diff = 1.0f - ambient;
    for (int f = 0; f < F; f++) {
        int a = faces[3*f], b = faces[3*f+1], c = faces[3*f+2];
        float x0 = px[a], y0 = py[a], x1 = px[b], y1 = py[b], x2 = px[c], y2 = py[c];
        float z0 = z[a], z1 = z[b], z2 = z[c];
        int minx = (int)floorf(fminf3(x0,x1,x2)), maxx = (int)ceilf(fmaxf3(x0,x1,x2));
        int miny = (int)floorf(fminf3(y0,y1,y2)), maxy = (int)ceilf(fmaxf3(y0,y1,y2));
        if (maxx < 0 || minx >= SW || maxy < 0 || miny >= SH) continue;
        if (minx < 0) minx = 0; if (maxx >= SW) maxx = SW-1;
        if (miny < 0) miny = 0; if (maxy >= SH) maxy = SH-1;
        float area = (x1-x0)*(y2-y0) - (x2-x0)*(y1-y0);
        if (fabsf(area) < 1e-6f) continue;
        float inv = 1.0f / area;
        // soft two-sided "half-lambert" headlight: gentle matte gradient, no
        // dark silhouette rim. shade in [ambient + diff*0.5, 1] -> high floor.
        float s0 = ambient + diff*(0.5f + 0.5f*fabsf(nz[a]));
        float s1 = ambient + diff*(0.5f + 0.5f*fabsf(nz[b]));
        float s2 = ambient + diff*(0.5f + 0.5f*fabsf(nz[c]));
        for (int yy = miny; yy <= maxy; yy++) {
            for (int xx = minx; xx <= maxx; xx++) {
                float pxc = xx + 0.5f, pyc = yy + 0.5f;
                float w0 = ((x1-pxc)*(y2-pyc) - (x2-pxc)*(y1-pyc)) * inv;
                float w1 = ((x2-pxc)*(y0-pyc) - (x0-pxc)*(y2-pyc)) * inv;
                float w2 = 1.0f - w0 - w1;
                if (w0 < 0 || w1 < 0 || w2 < 0) continue;
                float zz = w0*z0 + w1*z1 + w2*z2;
                int idx = yy*SW + xx;
                if (zz < zbuf[idx]) {
                    zbuf[idx] = zz;
                    float sh = w0*s0 + w1*s1 + w2*s2;
                    sc[3*idx]   = cr*sh; sc[3*idx+1] = cg*sh; sc[3*idx+2] = cb*sh;
                    cov[idx] = 1;
                }
            }
        }
    }

    // downsample ss x ss -> composite onto img with coverage-based alpha
    float inv_n = 1.0f / (float)(ss * ss);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float sr = 0, sg = 0, sb = 0; int cnt = 0;
            for (int dy = 0; dy < ss; dy++) {
                int sy = y*ss + dy;
                for (int dx = 0; dx < ss; dx++) {
                    int si = sy*SW + (x*ss + dx);
                    if (cov[si]) { sr += sc[3*si]; sg += sc[3*si+1]; sb += sc[3*si+2]; cnt++; }
                }
            }
            if (cnt == 0) continue;
            float coverage = cnt * inv_n;     // 0..1 at silhouette edges
            float a = alpha * coverage;
            float invc = 1.0f / cnt;
            int p = (y*W + x) * 3;
            float r = 255.f*sr*invc, g = 255.f*sg*invc, bl = 255.f*sb*invc;
            img[p]   = (unsigned char)((1.f-a)*img[p]   + a*(r  > 255.f ? 255.f : r));
            img[p+1] = (unsigned char)((1.f-a)*img[p+1] + a*(g  > 255.f ? 255.f : g));
            img[p+2] = (unsigned char)((1.f-a)*img[p+2] + a*(bl > 255.f ? 255.f : bl));
        }
    }

    free(z); free(px); free(py); free(nx); free(ny); free(nz);
    free(zbuf); free(sc); free(cov);
}
