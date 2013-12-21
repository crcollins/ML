from django.db import models

# Create your models here.

class Name(models.Model):
    name = models.CharField(max_length=300)

class StandardName(Name):
    left_name = models.CharField(max_length=50)
    core_name = models.CharField(max_length=3)
    middle_name = models.CharField(max_length=50)
    right_name = models.CharField(max_length=50)
    n = models.IntegerField()
    m = models.IntegerField()
    x = models.IntegerField()
    y = models.IntegerField()
    z = models.IntegerField()

    def full_name(self):
        nm = 'n%d_m%d' % (self.n, self.m)
        xyz = 'x%d_y%d_z%d' % (self.x, self.y, self.z)
        return '_'.join([self.left_name, self.core_name, self.middle_name, self.right_name, nm, xyz])


class DataPoint(models.Model):
    name = models.ForeignKey(Name, related_name="data")
    occupied = models.FloatField()
    virtual = models.FloatField()
    homo_orbital = models.IntegerField()
    energy = models.FloatField()
    dipole = models.FloatField()
    band_gap = models.FloatField()
