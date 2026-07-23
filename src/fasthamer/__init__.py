"""fasthamer — realtime HaMeR 3D hand mesh recovery on the Apple Neural Engine.

    import fasthamer

    hands = fasthamer.load(mode="video")      # or mode="image"
    result = hands(rgb_frame)                 # HandMeshResult
    for hand in result:
        hand.keypoints_2d      # (21, 2) pixels
        hand.keypoints_3d      # (21, 3) meters, hand-centered
        hand.cam_t             # (3,) hand -> camera translation
        hand.global_orient     # (3, 3) MANO global orientation
        hand.hand_pose         # (15, 3, 3) MANO pose (also .hand_pose_aa)
        hand.betas             # (10,) MANO shape
    overlay = hands.render(rgb_frame, result)
"""
from .rendering import MESH_COLOR, MeshRenderer, draw_landmarks
from .stabilize import HandednessStabilizer
from .tracker import HandMesh, load
from .types import HAND_EDGES, Hand, HandMeshResult

__version__ = "0.4.0"
__all__ = ["load", "HandMesh", "Hand", "HandMeshResult", "MeshRenderer",
           "draw_landmarks", "HandednessStabilizer", "HAND_EDGES", "MESH_COLOR",
           "__version__"]
