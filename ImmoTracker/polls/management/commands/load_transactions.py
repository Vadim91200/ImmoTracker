from django.core.management.base import BaseCommand
import pandas as pd
import geopandas as gpd
import json
from polls.models import Biens

class Command(BaseCommand):
    help = 'Load transactions data from txt files into the Database'

    def add_arguments(self, parser):
        parser.add_argument('files', nargs='+', type=str)

    def handle(self, *args, **options):
        for file_name in options['files']:
            self.stdout.write(f'Loading data from {file_name}...')
            # Charger les données
        df = pd.read_csv(file_name, delimiter = '|', low_memory=False,decimal=",",date_format='%d/%m/%Y',parse_dates=['Date mutation'])
        fr_dep = gpd.read_file("../asset/france-departements.geojson")
        fr_reg= gpd.read_file("../asset/france-regions.geojson")
        with open('../asset/france_regions.json', 'r') as f:
            fr_reg_code = json.load(f)
        df = df[df['Surface reelle bati'] != '0']
        df = df[df['Surface terrain'] != '0']
        #remove na and duplicates
        df=df.dropna(subset=['Date mutation', 'Valeur fonciere', 'Surface reelle bati', 'Code departement'])
        df=df.drop_duplicates(subset=['Date mutation','Nature mutation','Valeur fonciere','Type de voie','Code voie','Voie','Code postal','Commune','Code departement','Code departement','Code commune'])
        df.loc[:,'Date mutation'] = pd.to_datetime(df['Date mutation'], format='%d/%m/%Y')
        df = df[df['Surface reelle bati'] != 0]
        df = df[df['Surface terrain'] != 0]
        # filtrage sur les appartements et maisons
        biens=df[df['Type local'].str.contains("Appartement|Maison")]
        #ajout des codes regions
        biens['Code region']=biens['Code departement'].apply(lambda x: fr_reg_code[x])
        biens.loc[:,'m2'] = biens['Valeur fonciere'] / (biens['Surface terrain']+biens['Surface reelle bati'])
        for _, row in biens.iterrows():
            Biens.objects.get_or_create(
                Code_region=row['Code region'],
                Date_mutation=row['Date mutation'],
                Code_departement = row['Code departement'],
                Valeur_fonciere = row['Valeur fonciere'],
                Surface_terrain = row['Surface terrain'],
                m2 = row['m2']
            )
        self.stdout.write(f'Data from {file_name} loaded successfully')