from django.http import HttpResponse
from django.template import loader
from django.shortcuts import render
from .models import Biens
import geopandas as gpd
from shapely.geometry import shape
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import json
from plotly.offline import plot
from .forms import DepartementsForm
from django.core import serializers
from plotly.subplots import make_subplots
def home(request):
    return render(request, 'home.html')
def plot(request):
    file_name = "../Data/valeursfoncieres-2022.txt"
    df = pd.read_csv(file_name, delimiter='|', low_memory=False,
                     decimal=",", date_format='%d/%m/%Y', parse_dates=['Date mutation'])
    fr_dep = gpd.read_file("../asset/france-departements.geojson")
    fr_reg = gpd.read_file("../asset/france-regions.geojson")
    with open('../asset/france_regions.json', 'r') as f:
            fr_reg_code = json.load(f)
    df = df[df['Surface reelle bati'] != '0']
    df = df[df['Surface terrain'] != '0']
    # remove na and duplicatesFfF
    df = df.dropna(subset=['Date mutation', 'Valeur fonciere',
                       'Surface reelle bati', 'Code departement'])
    df = df.drop_duplicates(subset=['Date mutation', 'Nature mutation', 'Valeur fonciere','Type de voie','Code voie','Voie','Code postal','Commune','Code departement','Code departement','Code commune'])
    df.loc[:, 'Date mutation'] = pd.to_datetime(
            df['Date mutation'], format='%d/%m/%Y')
    df = df[df['Surface reelle bati'] != 0]
    df = df[df['Surface terrain'] != 0]
        # filtrage sur les appartements et maisons
    biens = df[df['Type local'].str.contains("Appartement|Maison")]
        # ajout des codes regions
    biens['Code region'] = biens['Code departement'].apply(
            lambda x: fr_reg_code[x])
    biens.loc[:, 'm2'] = biens['Valeur fonciere'] / \
            (biens['Surface terrain']+biens['Surface reelle bati'])
    mean_m2_dep = biens.groupby('Code departement')['m2'].mean(
    ).reset_index().rename(columns={'Code departement': 'code'})
    t = fr_dep.merge(mean_m2_dep, on='code')
    # centre de la France
    center_lat = 46.603354
    center_lon = 1.888334

    fig = px.choropleth_mapbox(t, geojson=fr_dep, locations='code', color='m2',
                               color_continuous_scale="ylgnbu",
                               range_color=(0, 25000),
                               mapbox_style="carto-positron",
                               featureidkey="properties.code",
                               zoom=4, center={"lat": center_lat, "lon": center_lon},
                               opacity=1,
                               hover_data=['nom', 'm2', 'code'],
                           labels={'m2': 'prix m2 moyen', 'code': 'Code departement', 'nom':'departement'})
    fig.update_layout(margin={"r": 0, "t": 0,"l":0,"b":0})
    MapBox = fig.to_html(
        full_html=False, default_height=500, default_width=700)
    # ScatterGosurfacemoyenneparregion
    mean_s_reg = biens.groupby('Code region')['Surface terrain'].mean(
    ).reset_index().rename(columns={'Code region': 'code'})
    m = fr_reg.merge(mean_s_reg, on='code').sort_values(
        'Surface terrain', ascending=False)
    m['centroid'] = m['geometry'].apply(lambda x: [shape(x).centroid])
    m['Text'] = m['nom'] + ' - Surface terrain: ' + \
        m['Surface terrain'].astype(str) + ' m2'

    fig = go.Figure()
    limits = [(0, 500), (501, 600),(601,700),(701,1000),(1001,3000)]
    colors = ["#5fad56", "#f2c14e", "#f78154","#4d9078","#b4436c"]
    for i in range(len(limits)):
        lim = limits[i]
        m_sub = m[(m['Surface terrain'] >= lim[0]) &
                   (m['Surface terrain'] <= lim[1])]
        fig.add_trace(go.Scattergeo(
            lat=m_sub['centroid'].apply(lambda x: x[0].y),
            lon=m_sub['centroid'].apply(lambda x: x[0].x),
            text=m_sub['Text'],
            featureidkey='properties.code',
            marker=dict(
                size=m_sub['Surface terrain'],
                color=colors[i],
                line_color='rgb(40,40,40)',
                line_width=0.5,
                sizemode='area',

            ),
            name='{0} - {1}'.format(lim[0], lim[1])))

    fig.update_geos(
        center=dict(lon=2, lat=46),
        projection_scale=12,
        projection_type="natural earth",
        landcolor='rgb(217, 217, 217)'
    )
    fig.update_layout(
        showlegend=True,
        margin={"r": 0, "t": 50,"l":0,"b":0},
    )
    ScatterGosurfacemoyenneparregion = fig.to_html(full_html=False, default_height=500, default_width=700)
    #ValeurFonciereBox
    mean_vf_reg=biens.groupby('Code region')['Valeur fonciere'].mean().reset_index().rename(columns={'Code region':'code'}).merge( fr_reg, on='code')
    fig = px.box(mean_vf_reg, y="Valeur fonciere",points='all',labels={'nom':'Region','code':'Code region','Valeur fonciere':'Valeur fonciere'},hover_data=['nom','Valeur fonciere','code'])
    ValeurFonciereBox = fig.to_html(
        full_html=False, default_height=500, default_width=700)

    # Scatter plot
    selected_departement = request.GET.get('departement')
    departement = biens[(biens['Code departement'] == selected_departement)]
    fig = px.scatter(departement, x="Surface reelle bati", y="Valeur fonciere", symbol='Type local', color='Type local',
                     size='Nombre pieces principales')
    DepartementScatter = fig.to_html(
        full_html=False, default_height=500, default_width=700)
    #CommunesBar
    departement=biens[(biens['Code departement']==selected_departement)].sort_values(by=['Valeur fonciere','Type local'])
    commune_departement = departement.groupby('Commune').agg({'Valeur fonciere':'sum', 'm2':'count'}).rename(columns={'Valeur fonciere': 'Somme des valeurs foncieres', 'm2': 'Nombres de biens vendus'}).reset_index()
    fig = px.bar(commune_departement, y="Somme des valeurs foncieres", x="Commune",text='Nombres de biens vendus')
    CommunesBar = fig.to_html(
        full_html=False, default_height=500, default_width=700)
    #DepartementNbPiècesPrincipal
    departement=departement.sort_values('Nombre pieces principales')
    fig = px.bar(departement, y="Surface terrain", x="Type local",color='Nombre pieces principales')
    DepartementNbPiècesPrincipal = fig.to_html(
        full_html=False, default_height=500, default_width=700)
    #Departementscatterm²
    departement=biens[(biens['Code departement']==selected_departement)]
    fig = px.scatter_3d(departement, x='m2', y='Surface reelle bati', z='Surface terrain',color='Type local')
    Departementscatterm2 = fig.to_html(
        full_html=False, default_height=500, default_width=700)
    #GoFigure
    fig = go.Figure(data=go.Heatmap(
        z=departement['Valeur fonciere'],
        x=departement['Date mutation'],
        y=departement['Nombre pieces principales'],
        colorscale='Viridis'))
    fig.update_layout(
        xaxis_title='Date mutation',
        yaxis_title='Nombre de pieces principales'
    )
    GoFigure = fig.to_html(
        full_html=False, default_height=500, default_width=700)
    #DepartementCenterPoint
    center_points=fr_dep.rename(columns={'code':'Code departement'})
    center_points['centroid'] = center_points['geometry'].apply(lambda x:shape(x).centroid)
    heated_data=biens.groupby('Code departement')['Valeur fonciere'].mean().reset_index()
    heated_data=center_points.merge(heated_data,on='Code departement')
    fig = go.Figure(go.Densitymapbox(lat=heated_data['centroid'].apply(lambda x: x.y), lon=heated_data['centroid'].apply(lambda x: x.x), z=heated_data['Valeur fonciere'],
                                 radius=10))

    fig.update_traces(
        colorscale='Viridis',
        opacity=0.8,
        zmin=0,
        zmax=1000000
    )

    fig.update_layout(mapbox_style="stamen-terrain", mapbox_center_lon=center_lon,mapbox_center_lat=center_lat,mapbox_zoom=4)
    fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
    DepartementCenterPoint = fig.to_html(
        full_html=False, default_height=500, default_width=700)
    # UltraMarins Histogram
    ultramarin=biens[(biens['Code region']=='01')|(biens['Code region']=='02')|(biens['Code region']=='03')|(biens['Code region']=='04')]
    departement_ultramarin=['Guadeloupe','Martinique','Guyane','La Reunion']
    ultramarin.loc[:,'Code region'] = ultramarin['Code region'].map(lambda x: departement_ultramarin[int(x) - 1])

    fig = px.histogram(ultramarin, x="Date mutation", y="Valeur fonciere", color="Type local",
                   marginal="box",
                   hover_data=ultramarin.columns)
    UltraMarinsHistogram = fig.to_html(
        full_html=False, default_height=500, default_width=700)
    #UltraMarinsSubplots
    ultramarine=ultramarin.groupby('Code region')['Valeur fonciere'].mean()
    fig = make_subplots(rows=2, cols=2)
    fig.add_trace(go.Histogram(histfunc="avg", y=ultramarin['Valeur fonciere'], x=ultramarin['Code region'], name="Valeur fonciere"),1,1)
    fig.add_trace(go.Histogram(histfunc="avg", y=ultramarin['Nombre pieces principales'], x=ultramarin['Code region'], name="Nombre pieces principales"),1,2)
    fig.add_trace(go.Histogram(histfunc="avg", y=ultramarin['Surface reelle bati'], x=ultramarin['Code region'], name="Surface reelle bati"),2,1)
    fig.add_trace(go.Histogram(histfunc="avg", y=ultramarin['Surface terrain'], x=ultramarin['Code region'], name="Surface terrain"),2,2)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='text',
                         showlegend=True, legendgroup='Additional', name='Code region'), row=1, col=1)

    fig.update_layout(
        xaxis=dict(title='Valeurs foncières'),
        xaxis2=dict(title='Nombre de pièces'),
        xaxis3=dict(title='Surface du terrain'),
        xaxis4=dict(title='Surface réelle bâtie'),
        showlegend=True,
        legend=dict(x=1.05, y=0.5)
    )
    UltraMarinsSubplots = fig.to_html(
        full_html=False, default_height=500, default_width=700)
    #RegionbyDayLine
    selected_region = request.GET.get('region')
    Regionbyday=biens[(biens['Code region']==selected_region)].groupby(['Date mutation','Code departement']).agg({'m2':'mean'}).reset_index().dropna().sort_values(by=['Date mutation'])
    fig = px.line(Regionbyday, x="Date mutation", y="m2",color='Code departement')
    RegionbyDayLine = fig.to_html(
        full_html=False, default_height=500, default_width=700)
    #sunburstParisMarseilleLyon
    df_paris_marseille_lyon = df[
    df['Type local'].isin(['Appartement', 'Maison']) &
    df['Code postal'].astype(str).str.startswith(('750', '1300', '1301', '690')) &
    df['Code departement'].isin(['75', '13', '69'])]

    df_paris_marseille_lyon.loc[:, 'Code departement'] = df_paris_marseille_lyon['Code departement'].replace('75', 'Paris').replace('13', 'Marseille').replace('69', 'Lyon')


    df_paris_marseille_lyon.loc[:, 'Surface carrez total'] = df_paris_marseille_lyon[
        ['Surface Carrez du 1er lot',
        'Surface Carrez du 2eme lot',
        'Surface Carrez du 3eme lot',
        'Surface Carrez du 4eme lot',
        'Surface Carrez du 5eme lot']].sum(axis=1)

    moy_df = df_paris_marseille_lyon.groupby(['Code departement', 'Commune'])['Surface carrez total'].mean().reset_index()
# moy_df.loc[:, 'to sort'] = moy_df['Commune'].apply(lambda x: x.replace('EME', '').replace('ER', ''))\n",
# moy_df[['str', 'num']] = moy_df['to sort'].str.split(' ', expand=True)
# moy_df['num'] = moy_df['num'].astype(float)
# moy_df.sort_values(['str', 'num'], inplace=True)

    moy_df.rename(columns={'Surface carrez total': 'Surface carrez moy'}, inplace=True)


    fig = px.sunburst(moy_df,
                  path=[px.Constant('Moyenne Totale'), 'Code departement', 'Commune'],
                  values='Surface carrez moy',
                  maxdepth=2,
                  color='Surface carrez moy')
    sunburstParisMarseilleLyon = fig.to_html(
        full_html=False, default_height=500, default_width=700)
    #subplot4regions
    df_maison_appart_commerce = df[df['Type local'].isin(['Appartement', 'Maison', 'Local industriel. commercial ou assimilé'])]
    df_idf = df_maison_appart_commerce[df_maison_appart_commerce['Code departement'].map(fr_reg_code) == '11']
    df_bretagne = df_maison_appart_commerce[df_maison_appart_commerce['Code departement'].map(fr_reg_code) == '53']
    df_cote_azure = df_maison_appart_commerce[df_maison_appart_commerce['Code departement'].map(fr_reg_code) == '93']
    df_rhone_alpe = df_maison_appart_commerce[df_maison_appart_commerce['Code departement'].map(fr_reg_code) == '84']

    fig = make_subplots(rows=2, cols=2, specs=[[{'type': 'pie'}, {'type': 'pie'}], [{'type': 'pie'}, {'type': 'pie'}]],
                    subplot_titles=['Ile de France',
                                    'Bretagne',
                                    'Provence-Alpes-Côte d\'Azur',
                                    'Auvergne-Rhône-Alpes'])
    fig.add_trace(go.Pie(labels=df_idf['Type local']), 1, 1)
    fig.add_trace(go.Pie(labels=df_bretagne['Type local']), 1, 2)
    fig.add_trace(go.Pie(labels=df_cote_azure['Type local']), 2, 1)
    fig.add_trace(go.Pie(labels=df_rhone_alpe['Type local']), 2, 2)

    subplot4regions = fig.to_html(
        full_html=False, default_height=500, default_width=700)
    #iciclevaleurfoncierniveaux
    with open('../asset/region_name.json', 'r') as f:
        region_name = json.load(f)
    with open('../asset/departement_name.json', 'r') as f:
        departement_name = json.load(f)

    df_filtre = df.copy()
    df_filtre = df_filtre[df_filtre['Type local'].isin(['Appartement', 'Maison'])]
    f_filtre = df_filtre[(df_filtre['Valeur fonciere'] > 0) & (df_filtre['Valeur fonciere'] <= 1000000)]

    df_filtre.loc[:,'Code region'] = df_filtre['Code departement'].map(fr_reg_code)
    df_filtre = df_filtre[~df_filtre['Code region'].isin(['01', '02', '03', '04', '05', '06', '94'])]
    df_filtre['Code region'] = df_filtre['Code region'].map(region_name).str.encode('iso-8859-1')
    df_filtre['Code region'] = df_filtre['Code region'].astype(str)

    df_filtre['Code departement'] = df_filtre['Code departement'].map(departement_name).str.encode('iso-8859-1')

    df_grouped = df_filtre.groupby(['Code region', 'Code departement', 'Commune', 'Code postal', 'Code voie', 'Voie']).first()
    df_moy = df_grouped.groupby(['Code region', 'Code departement', 'Commune'])['Valeur fonciere'].mean().reset_index()

    fig = px.icicle(df_moy,
                path=[px.Constant('Total moyenne'), 'Code region', 'Code departement', 'Commune'],
                values='Valeur fonciere',
                color='Valeur fonciere',
                maxdepth=2)
    iciclevaleurfoncierniveaux = fig.to_html(
        full_html=False, default_height=500, default_width=700)
    #SunburstVentebien2022
    df_date = df_filtre.copy(deep=True)['Date mutation'].to_frame()
    df_date.loc[:,'Mois'] = df_date['Date mutation'].dt.month
    df_date.loc[:,'Jours'] = df_date['Date mutation'].dt.day
    mois_dict = {
        1: "Janvier",
        2: "Février",
        3: "Mars",
        4: "Avril",
        5: "Mai",
        6: "Juin",
        7: "Juillet",
        8: "Août",
        9: "Septembre",
        10: "Octobre",
        11: "Novembre",
        12: "Décembre"
    }
    df_date.loc[:, 'Mois'] = df_date['Mois'].map(mois_dict)
    df_date_groupby = df_date.groupby(['Mois', 'Jours']).count().reset_index().rename(columns={'Date mutation': 'Total'})
    fig = px.sunburst(df_date_groupby,
                  path=[px.Constant('Total'), 'Mois', 'Jours'],
                  values='Total',
                  color='Total',
                  maxdepth=2)
    SunburstVentebien2022 = fig.to_html(
        full_html=False, default_height=500, default_width=700)
    #ClassementVenteRegion
    df_region = df_filtre[['Code region', 'Type local']]
    df_region.loc[:, 'Maison'] = df_region['Type local'] == 'Maison'
    df_region.loc[:, 'Appartement'] = df_region['Type local'] == 'Appartement'

    df_region_groupby = df_region.groupby(['Code region']).agg({'Maison':'sum', 'Appartement':'sum'}).reset_index()

    df_region_groupby.loc[:,'Total'] = df_region_groupby['Maison'] + df_region_groupby['Appartement']
    df_region_groupby = df_region_groupby.sort_values('Total')

    fig = px.bar(df_region_groupby, x=['Maison', 'Appartement'], y='Code region', orientation='h')
    ClassementVenteRegion = fig.to_html(
        full_html=False, default_height=500, default_width=700)
    #EtudeDepensesRegion
    df_valeur = df_filtre[['Code region', 'Valeur fonciere']]

    df_valeur_groupby = df_valeur.groupby(['Code region']).sum().reset_index()
    df_valeur_groupby = df_valeur_groupby.sort_values('Valeur fonciere')

    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'bar'}, {'type': 'pie'}]],
                    subplot_titles=['Classement des dépenses foncieres',
                                    'Repartition des dépenses foncieres'])

    fig.add_trace(go.Bar(
        x=df_valeur_groupby['Valeur fonciere'],
        y=df_valeur_groupby['Code region'],
        orientation='h',
        marker=dict(color=df_valeur_groupby['Valeur fonciere'])), 1, 1)

    fig.add_trace(go.Pie(labels=df_valeur_groupby['Code region'],
                     values=df_valeur_groupby['Valeur fonciere'],
                     textinfo='label+percent',
                     marker=dict(colors=df_valeur_groupby['Valeur fonciere'])), 1, 2)
    fig.update_layout(
        showlegend=False,
    )
    EtudeDepensesRegion = fig.to_html(
        full_html=False, default_height=500, default_width=700)
    #EtudeSurfaces
    carrez_col = ['Surface Carrez du 1er lot',
     'Surface Carrez du 2eme lot',
     'Surface Carrez du 3eme lot',
     'Surface Carrez du 4eme lot',
     'Surface Carrez du 5eme lot']


    df_proportion = df_filtre[['Nombre pieces principales', 'Surface reelle bati', *carrez_col]]
    df_proportion = df_proportion[(df_proportion['Nombre pieces principales'] <= 13) & (df_proportion['Nombre pieces principales'] > 0)]
    to_drop = df_proportion[(df_proportion[carrez_col].isnull()).all(axis=1)].index
    df_proportion.drop(to_drop, inplace=True)

    df_proportion.loc[:, 'Surface carrez total'] = df_proportion[carrez_col].sum(axis=1)

    df_proportion = df_proportion.drop(carrez_col, axis=1)
    df_proportion.loc[:,'Ratio'] =  df_proportion['Surface reelle bati'] / df_proportion['Surface carrez total']
    df_proportion = df_proportion.sort_values('Nombre pieces principales')

    df_proportion_grouped = df_proportion.groupby('Nombre pieces principales')\
    .agg({'Surface carrez total':'mean', 'Surface reelle bati':'mean', 'Ratio':'mean'}).reset_index()
    df_proportion_grouped = df_proportion_grouped.rename(columns={'Surface reelle bati': 'Surface reelle bati Moyenne', 'Surface carrez total': 'Surface carrez Moyenne'})


    fig = make_subplots(rows=1, cols=2,
                    subplot_titles=['Evolution des surfaces habitable et bati selon le nombre de piece',
                                    'Ratio surfaces bati/habitable selon le nombre de piece'])

    fig.add_trace(go.Scatter(
    x=df_proportion_grouped['Nombre pieces principales'],
    y=df_proportion_grouped['Surface carrez Moyenne'],
    mode='lines+markers',
    name='Surface carrez Moyenne'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
    x=df_proportion_grouped['Nombre pieces principales'],
    y=df_proportion_grouped['Surface reelle bati Moyenne'],
    mode='lines+markers',
    name='Surface reelle bati Moyenne'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
    x=df_proportion_grouped['Nombre pieces principales'],
    y=df_proportion_grouped['Ratio'],
    mode='lines+markers',
    name='Ratio'
    ), row=1, col=2)
    EtudeSurfaces = fig.to_html(
        full_html=False, default_height=500, default_width=700)
    #SommePiecesPrincipalesRegionDepartement5Communes
    df_piece = df_filtre[['Code region', 'Code departement', 'Commune', 'Nombre pieces principales']]


    def select_first_5(group):
        if len(group) <= 5:
            return group
        else:
            group = group.sort_values('Nombre pieces principales', ascending=False)
            sum_row = group.iloc[5:].groupby(['Code region', 'Code departement'])[
            'Nombre pieces principales'].mean().reset_index()
            sum_row['Nombre pieces principales'] = sum_row['Nombre pieces principales'].round()
            sum_row['Commune'] = 'Moyenne des autres'
        return pd.concat([group.iloc[:5], sum_row]).reset_index(drop=True)


    df_piece_groupby = df_piece.groupby(['Code region', 'Code departement', 'Commune']) \
        .agg({'Nombre pieces principales': 'sum'}).reset_index() \
        .groupby(['Code region', 'Code departement']) \
        .apply(select_first_5).reset_index(drop=True)

    fig = px.treemap(df_piece_groupby,
                 path=[px.Constant('Total'), 'Code region', 'Code departement', 'Commune'],
                 values='Nombre pieces principales',
                 maxdepth=2,
                 color='Nombre pieces principales')
    SommePiecesPrincipalesRegionDepartement5Communes = fig.to_html(
        full_html=False, default_height=500, default_width=700)
    context = {
        'MapBox': MapBox,
        'ScatterGosurfacemoyenneparregion': ScatterGosurfacemoyenneparregion,
        'ValeurFonciereBox': ValeurFonciereBox,
        'DepartementScatter': DepartementScatter,
        'CommunesBar': CommunesBar,
        'DepartementNbPiècesPrincipal': DepartementNbPiècesPrincipal,
        'Departementscatterm2': Departementscatterm2,
        'GoFigure': GoFigure,
        'DepartementCenterPoint': DepartementCenterPoint,
        'UltraMarinsHistogram': UltraMarinsHistogram,
        'UltraMarinsSubplots': UltraMarinsSubplots,
        'RegionbyDayLine': RegionbyDayLine,
        'sunburstParisMarseilleLyon': sunburstParisMarseilleLyon,
        'subplot4regions': subplot4regions,
        'iciclevaleurfoncierniveaux': iciclevaleurfoncierniveaux,
        'SunburstVentebien2022': SunburstVentebien2022,
        'ClassementVenteRegion': ClassementVenteRegion,
        'EtudeDepensesRegion': EtudeDepensesRegion,
        'EtudeSurfaces': EtudeSurfaces,
        'SommePiecesPrincipalesRegionDepartement5Communes': SommePiecesPrincipalesRegionDepartement5Communes,
        'form': DepartementsForm()
    }
    return render(request, 'plot.html', context)

def comparaisonplot(request):
    url2022 = "../Data/valeursfoncieres-2022.txt"
    url2019 = "../Data/valeursfoncieres-2019.txt"
    # centre de la France
    center_lat = 46.603354
    center_lon = 1.888334
    df2022 = pd.read_csv(url2022, delimiter = '|', low_memory=False,decimal=",",date_format='%d/%m/%Y',parse_dates=['Date mutation'])
    df2019 = pd.read_csv(url2019, delimiter = '|', low_memory=False,decimal=",",date_format='%d/%m/%Y',parse_dates=['Date mutation'])
    fr_dep = gpd.read_file("../asset/france-departements.geojson")
    fr_reg= gpd.read_file("../asset/france-regions.geojson")
    with open('../asset/france_regions.json', 'r') as f:
        fr_reg_code = json.load(f)
    with open('../asset/region_name.json', 'r') as f:
        region_name = json.load(f)
    with open('../asset/departement_name.json', 'r') as f:
        departement_name = json.load(f)
    def clean(df):
        df=df.dropna(subset=['Date mutation', 'Valeur fonciere', 'Surface reelle bati', 'Code departement'])
        df['Surface terrain']=df['Surface terrain'].fillna(0)
        df = df.drop(df[df['Surface reelle bati'] == 0].index)
        return df.drop_duplicates(subset=['Date mutation','Nature mutation','Valeur fonciere','Type de voie','Code voie','Voie','Code postal','Commune','Code departement','Code departement','Code commune'])
    df2022=clean(df2022)
    df2019=clean(df2019)
    # filtrage sur les appartements et maisons
    biens2022=df2022[df2022['Type local'].str.contains("Appartement|Maison")]
    biens2019=df2019[df2019['Type local'].str.contains("Appartement|Maison")]
    #ajout des codes regions
    biens2022.loc[:,'Code region'] = biens2022['Code departement'].apply(lambda x: fr_reg_code[x])
    biens2019.loc[:,'Code region'] = biens2019['Code departement'].apply(lambda x: fr_reg_code[x])

    #ValeurFonciere20192022
    biens2019.loc[:,'m2'] = biens2019['Valeur fonciere'] / (biens2019['Surface terrain']+biens2019['Surface reelle bati'])
    df2019_2022 = pd.concat([biens2019, biens2022], ignore_index=True)
    df2019_2022['Annee']=df2019_2022['Date mutation'].dt.year
    df2019_2022['Mois']=df2019_2022['Date mutation'].dt.month
    df2019_2022['Jour']=df2019_2022['Date mutation'].dt.day

    overview=df2019_2022.groupby('Annee').agg({'Valeur fonciere':'sum','Date mutation':'count','Surface reelle bati':'sum','m2':'mean'}).reset_index()
    overview

    fig = make_subplots(rows=2, cols=2)

    fig.add_trace(go.Bar(
        x=overview['Annee'],
        y=overview['Valeur fonciere'],
        name='Total des valeurs foncieres'),
        row=1,col=1
    )


    fig.add_trace(go.Bar(
        x=overview['Annee'],
        y=overview['Date mutation'],
        name='Nombre de ventes'),
        row=1,col=2
    )
    fig.add_trace(go.Bar(
        x=overview['Annee'],
        y=overview['Surface reelle bati'],
        name='Nombres de pieces principales moyennes'),
        row=2,col=1
    )
    fig.add_trace(go.Bar(
        x=overview['Annee'],
        y=overview['m2'],
        name='Prix moyen au m2'),
        row=2,col=2
    )
    fig.update_layout(hovermode="x unified")
    ValeurFonciere20192022 = fig.to_html(
        full_html=False, default_height=500, default_width=700)
    #SommeValueurFonciere20192022
    bydate=df2019_2022.groupby(['Mois','Annee']).agg({'Valeur fonciere':'sum'}).reset_index().sort_values(['Mois','Annee'],ascending=False)
    bydate['fake']=pd.to_datetime({'year': 2022, 'month': bydate['Mois'], 'day': 1})
    fig = px.line(bydate, x='fake', y='Valeur fonciere',hover_data=['fake', 'Valeur fonciere'], color='Annee')
    fig.update_xaxes(title='Date')
    fig.update_yaxes(title='Valeur fonciere')
    fig.update_traces(hovertemplate='')
    SommeValueurFonciere20192022 = fig.to_html(
        full_html=False, default_height=500, default_width=700)
    #Totalvaleurfonciereparregion
    sum_vf_reg= df2019_2022.groupby(['Code region','Annee'])['Valeur fonciere'].sum().reset_index().rename(columns={'Code region':'code'}).merge( fr_reg, on='code')

    fig = px.box(sum_vf_reg, y="Valeur fonciere",points='all',labels={'nom':'Region','code':'Code region','Valeur fonciere':'Valeur fonciere'},hover_data=['nom','Valeur fonciere','code'],color='Annee')
    Totalvaleurfonciereparregion = fig.to_html(
        full_html=False, default_height=500, default_width=700)
    #DispersionValeursFoncieresDepartementM2
    compare_m2vf=df2019_2022.groupby(['Code departement','Annee']).agg({'Valeur fonciere':'mean','m2':'mean'}).reset_index()
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=compare_m2vf[compare_m2vf['Annee']==2019]['Valeur fonciere'],
        y=compare_m2vf[compare_m2vf['Annee']==2019]['m2'],
        mode='markers',
        name='2019',
        marker=dict(
            symbol='x',
            opacity=0.7,
            color='white',
            size=8,
            line=dict(width=1),
        )
    ))
    fig.add_trace(go.Scatter(
        x=compare_m2vf[compare_m2vf['Annee']==2022]['Valeur fonciere'],
        y=compare_m2vf[compare_m2vf['Annee']==2022]['m2'],
        mode='markers',
        name='2022',
        marker=dict(
            symbol='circle',
            opacity=0.7,
            color='white',
            size=8,
            line=dict(width=1),
        )
    ))

    fig.add_trace(go.Histogram2d(
        x=compare_m2vf['Valeur fonciere'],
        y=compare_m2vf['m2'],
        colorscale='YlGnBu',
        showscale=False,
        zmax=16,
        nbinsx=14,
        nbinsy=14,
        zauto=False,
        hovertemplate='Valeur fonciere moyenne: %{x}<br>Prix m2 moyen: %{y}<br>Nombre de departements: %{z}<extra></extra>'
    ))
    fig.update_layout(
        xaxis=dict( ticks='', showgrid=False, zeroline=False, nticks=20 ,title='Valeur fonciere'),
        yaxis=dict( ticks='', showgrid=False, zeroline=False, nticks=20,title='prix/m2' ),
        hovermode='closest'
    )
    DispersionValeursFoncieresDepartementM2 = fig.to_html(
        full_html=False, default_height=500, default_width=700)
    #DifferencePrixM220192022
    compare_month=df2019_2022.groupby(['Code departement','Annee']).agg({'m2':'mean'}).reset_index()
    dep_diff=compare_month[compare_month['Annee']==2022]['Code departement'].reset_index().rename(columns={'Code departement':'code'})
    dep_diff['m2_diff']=compare_month[compare_month['Annee']==2022]['m2'].values-compare_month[compare_month['Annee']==2019]['m2'].values
    dep_diff = fr_dep.merge(dep_diff, on='code')
    fig = px.choropleth_mapbox(dep_diff, geojson=fr_dep, locations='code', color='m2_diff',
                           color_continuous_scale="portland",
                           range_color=(-4100,2900),
                           mapbox_style="carto-positron",
                           featureidkey="properties.code",
                           zoom=4, center = {"lat": center_lat, "lon": center_lon},
                           opacity=1,
                           hover_data=['nom','m2_diff','code'],
                           labels={'m2_diff':'Augmentation du prix m2','code':'code departement', 'nom':'departement'})
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    DifferencePrixM220192022 = fig.to_html(
        full_html=False, default_height=500, default_width=700)
    context = {
        'ValeurFonciere20192022': ValeurFonciere20192022,
        'SommeValueurFonciere20192022': SommeValueurFonciere20192022,
        'Totalvaleurfonciereparregion': Totalvaleurfonciereparregion,
        'DispersionValeursFoncieresDepartementM2': DispersionValeursFoncieresDepartementM2,
        'DifferencePrixM220192022': DifferencePrixM220192022
    }
    return render(request, 'comparaisonplot.html', context)
def about(request):
    return render(request, 'about.html')