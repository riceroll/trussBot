
def readObj(self, filename='data/cube.obj'):
    # depreciated
    mesh = pv.read(filename)
    tet = tetgen.TetGen(mesh)
    tet.make_manifold()
    tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5)
    eSet = set()
    for elem in tet.elem:
        elem = np.append(elem, elem[0])
        ij = [[i, j] for i in range(4) for j in range(4) if i != j]
        for i, j in ij:
            if (elem[i], elem[j]) in eSet or (elem[j], elem[i]) in eSet:
                pass
            else:
                eSet.add((elem[i], elem[j]))
    e = list(eSet)
    self.e = np.array([list(tu) for tu in e]).astype(np.int)
    self.v = np.array(tet.node)
    self.init()
