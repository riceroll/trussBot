import numpy as np
import open3d as o3     # open3d 0.10.0


class Viewer(object):
    def __init__(self, env=None):
        self.env = env
        self.model = env.model

        self.vis = o3.visualization.VisualizerWithKeyCallback()
        self.opt = None
        self.view = None

        self.lines = None

    def reset(self):
        import open3d as o3
        self.vis = o3.visualization.VisualizerWithKeyCallback()
        self.vis.create_window()
        self.opt = self.vis.get_render_option()
        self.view = self.vis.get_view_control()

        # render & view options
        self.opt.background_color = np.asarray([0.15, 0.15, 0.15])

    # draw
    def drawGround(self):
        n = 20
        vs = []
        es = []
        for i, x in enumerate(np.arange(1 - n, n)):
            vs.append([x, 1 - n, 0])
            vs.append([x, n - 1, 0])
            es.append([i * 2, i * 2 + 1])
        lines = o3.geometry.LineSet(points=o3.utility.Vector3dVector(vs), lines=o3.utility.Vector2iVector(es))
        self.vis.add_geometry(lines)

        vs = []
        es = []
        for i, x in enumerate(np.arange(1 - n, n)):
            vs.append([1 - n, x, 0])
            vs.append([n - 1, x, 0])
            es.append([i * 2, i * 2 + 1])
        lines = o3.geometry.LineSet(points=o3.utility.Vector3dVector(vs), lines=o3.utility.Vector2iVector(es))
        self.vis.add_geometry(lines)

    def drawAxis(self):
        v = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        e = [[0, 1], [0, 2], [0, 3]]
        c = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        ground = o3.geometry.LineSet(points=o3.utility.Vector3dVector(v), lines=o3.utility.Vector2iVector(e))
        ground.colors = o3.utility.Vector3dVector(c)
        self.vis.add_geometry(ground)

    def drawMesh(self):
        self.lines = o3.geometry.LineSet(points=o3.utility.Vector3dVector(self.model.v),
                                         lines=o3.utility.Vector2iVector(self.model.e))
        self.lines.colors = o3.utility.Vector3dVector(np.ones([self.model.e.shape[0], 3]))
        self.vis.add_geometry(self.lines)

    def drawAll(self):
        self.drawGround()
        self.drawAxis()
        self.drawMesh()

        self.view.rotate(0, -400)
        self.view.scale(100)
        self.view.set_constant_z_far(10000)

    # update
    def updateMesh(self):
        for i, v in enumerate(self.model.v):
            self.lines.points[i] = v
        self.vis.update_geometry(self.lines)

    def updateAll(self):
        self.updateMesh()

    # routine
    def timerCallback(self, vis):
        action = self.env.getAction()
        self.env.step(action)
        self.updateAll()

    def registerAnimationCallback(self):
        self.vis.register_animation_callback(self.timerCallback)

    # key callbacks
    def keyCallbackActuation(self, vis, action, mod):
        if action:
            self.model.switch()
            print('pressure: ', self.model.pressure)
            pass

    def keyCallbackPause(self, vis, action, mod):
        if action:
            self.model.pause = not self.model.pause
            print('pause: ', self.model.pause)
            pass

    def registerKeyCallback(self):
        callbacks = {
            65: self.keyCallbackActuation,   # a
            80: self.keyCallbackPause,       # p
        }

        for code in callbacks:
            self.vis.register_key_action_callback(code, callbacks[code])


    def run(self):
        self.vis.run()
        self.vis.destroy_window()
