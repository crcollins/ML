from django.db import models

# Create your models here.

class Name(models.Model):
    pass

class StandardName(Name):
    left_name = models.CharField()
    core_name = models.CharField()
    middle_name = models.CharField()
    right_name = models.CharField()
    n = models.IntegerField()
    m = models.IntegerField()
    x = models.IntegerField()
    y = models.IntegerField()
    z = models.IntegerField()

    def full_name(self):
        nm = 'n%d_m%d' % (self.n, self.m)
        xyz = 'x%d_y%d_z%d' % (self.x, self.y, self.z)
        return '_'.join([self.left_name, self.core_name, self.middle_name, self.right_side, nm, xyz])


class DataPoint(models.Model):
    occupied = models.FloatField()
    virtual = models.FloatField()
    homo_orbital = models.IntegerField()
    energy = models.FloatField()
    dipole = models.FloatField()
    band_gap = models.FloatField()
