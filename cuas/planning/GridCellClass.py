#GridCellClass.py
class GridCell:
    """
    Define grid cell class in SRM's grid
    """
    def __init__(self, x, dx, dy, level):

        self.x_ = x
        self.dx_ = dx
        self.dy_ = dy
        self.level_ = level
        self.particles_ = []

    def addParticle(self, particle):
        self.particles_.append(particle)



