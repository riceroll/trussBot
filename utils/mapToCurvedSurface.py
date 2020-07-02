import numpy as np
# import open3d as o3d


def mapCurve(vs):
# map vertices of the model to a curved surface
  R = 4
  d = -0.6
  ps = []

  for v in vs:
    x, y, z = v
    alpha = (x - 4) / R
    xx = 4 + R * np.sin(alpha)
    zz = R + d - R * np.cos(alpha)
    if z == 1:
        vec = np.array([4-xx,0,R+d-zz])
        vec = vec / np.sqrt(np.sum(vec**2))
        xx = xx + vec[0]
        zz = zz + vec[2]
    ps.append(np.array([xx, y, zz]))

  # pcd = o3d.geometry.PointCloud()
  # pcd.points = o3d.utility.Vector3dVector(ps)
  # o3d.visualization.draw_geometries([pcd])

  return ps
