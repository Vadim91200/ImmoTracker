from django.db import models

class Biens(models.Model):
    Code_region= models.CharField(max_length=100)
    Date_mutation= models.CharField(max_length=100)
    Code_departement = models.CharField(max_length=100)
    Valeur_fonciere = models.CharField(max_length=100)
    Surface_terrain = models.CharField(max_length=100)
    m2 = models.CharField(max_length=100)

    def __str__(self):
        return str(self.date_mutation)

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)