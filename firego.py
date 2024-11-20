from shapely.geometry import Point, LineString
import geopandas as gpd
import pandas as pd
from shapely.affinity import scale, rotate
import math, random, tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import numpy as np

class FireGo:
    def __init__(self, building, seasons_attr, stop_gdf, firepro_max=0.5):
        def _get_neighbors(gdf, id):  # fire spread calculation without wind direction and wind speed
            d = 50
            x = gdf[gdf['id'] == id]['centroid'].iloc[0].x
            y = gdf[gdf['id'] == id]['centroid'].iloc[0].y

            buffer = Point(x,y).buffer(d)
            buffer_gdf = gpd.GeoDataFrame(geometry=[buffer], crs=gdf.crs)

            # all buildings in buffer
            nearby_buildings = gdf[gdf.intersects(buffer_gdf.unary_union)]
            intersecting_ids = nearby_buildings['id'].tolist()
            return intersecting_ids

        if 'id' not in building.columns:
            raise ValueError('building must have an "id" column')
        
        self._gdf = building.copy()
        self._gdf['area'] = self._gdf.area
        self._gdf['centroid'] = self._gdf.centroid
        self.stop_geoms = stop_gdf['geometry']
        self.seasons_attr = seasons_attr
        nearids = self._gdf[['id', 'centroid', 'geometry']].copy()
        nearids['near_id'] = nearids['id'].apply(lambda x: _get_neighbors(nearids, x))
        self.near_id = nearids
        self.season_near_id = None
        self.firepro_max = firepro_max
        self.mapping = {'type1': ['起火','#9ACD32'],  # ignition
                    'type2': ['蔓延','#87CEEB'],  # flashover
                    'type3': ['完全燃烧','#9400D3'],  # full development
                    'type4': ['倒塌','#FF8C00'],  # structure fire
                    'type5': ['熄灭','#191970']}  # collapse

    def _i_ellipse(self, wind_angle, wind_speed, build_row):
        d = math.sqrt(build_row.area.iloc[0])  # square root of building area
        normal_radius = 3 + d/2
        wind_radius = normal_radius + wind_speed  # expanded radius
        radians = math.radians(wind_angle)  # rotation radians with wind direction
        long_axis = normal_radius+wind_radius
        x = build_row.centroid.x.iloc[0] + math.cos(radians)*long_axis
        y = build_row.centroid.y.iloc[0] + math.sin(radians)*long_axis
        circle = Point(x,y).buffer(1)
        ellipse = scale(circle, xfact = 2*wind_radius, yfact = 2*normal_radius)
        ellipse = rotate(ellipse, wind_angle, origin=(x,y))
        # create gdf
        ellipse_gdf = gpd.GeoDataFrame(geometry=[ellipse], crs=self._gdf.crs)
        # normal_circle  = Point(build_row.centroid.x, build_row.centroid.y).buffer(normal_radius)
        # normal_gdf = gpd.GeoDataFrame(geometry=[normal_circle], crs=gdf.crs)
        return ellipse_gdf, long_axis
    
    def _get_radius_neighbors(self, gdf, id):  # fire spread calculation based on wind direction and wind speed
        firebuild = gdf[gdf['id']==id].copy()  # id of burning building
        ellipse_gdf, _ = self._i_ellipse(self.wind_angle, self.wind_speed, firebuild)
        # all buildings in buffer
        nearby_buildings = gdf[gdf.intersects(ellipse_gdf.unary_union)]
        intersecting_ids = nearby_buildings['id'].tolist()
        return intersecting_ids
    
    def _firestop(self, id_x, id_y):
        centroid1 = self._gdf[self._gdf['id']==id_x].centroid
        centroid2 = self._gdf[self._gdf['id']==id_y].centroid

        # extract coords of centroid
        centroid1_coords = (centroid1.iloc[0].x, centroid1.iloc[0].y)
        centroid2_coords = (centroid2.iloc[0].x, centroid2.iloc[0].y)

        # create a segment-line connecting the two centroids
        connecting_line = LineString([centroid1_coords, centroid2_coords])
        
        # check whether the segment-line intersects with walls, rivers, roads
        for i_stop in self.stop_geoms:
            intersect = connecting_line.intersection(i_stop).length
            if intersect!=0:
                return intersect
                break
            return 0

    def _firepro(self, building, fire_id, neibor_id):
        status = building[building['id']==fire_id]['status'].iloc[0]
        if status == 'type3':
            p_status = 0.4 
        else:
            p_status = 1.0

        fire_bd = self._gdf[self._gdf['id']==fire_id].copy()
        neibor_bd = self._gdf[self._gdf['id']==neibor_id].copy()
        building_i = fire_bd.centroid.iloc[0].coords[0]
        building_j = neibor_bd.centroid.iloc[0].coords[0]
        fire_ellipse, long_axis = self._i_ellipse(self.wind_angle, self.wind_speed, fire_bd)

        # direction vector of the line
        direct = np.array([building_j[0] - building_i[0], building_j[1] - building_i[1]])  
        direct = direct / np.linalg.norm(direct)  # 单位向量  

        # extension line  
        extended_start = building_j  # extend from building j
        extended_end = (extended_start[0] + direct[0] * long_axis,  
                        extended_start[1] + direct[1] * long_axis)  

        extended_line = LineString([extended_start, extended_end])  
        intersection = extended_line.intersection(fire_ellipse) 

        dis_ij = fire_bd.geometry.iloc[0].distance(neibor_bd.geometry.iloc[0])
        dis_ib = intersection.length + dis_ij

        p_dis = 1 - dis_ij/dis_ib
        materiallist = {'wood':1,'wood & concrete':0.8,'concrete':0.6}
        pro = p_dis * p_status * materiallist[neibor_bd['material'].iloc[0]]

        if dis_ij > 0:
            stop_lenth = self._firestop(fire_id, neibor_id)
            stop_coef = stop_lenth/dis_ij
            pro *= (1-stop_coef)

        return pro 
    
    def _status(self, time, mat, seed=42):
        # random time effect
        random.seed(seed)
        if mat == 'wood':
            t1,t2 = random.randint(4,6), random.randint(5,8)
            t3,t4 = random.randint(10,20), random.randint(20,30)
        elif mat == 'wood & concrete':
            t1,t2 = random.randint(4,6), random.randint(5,8)
            t3,t4 = random.randint(20,30), random.randint(30,40)
        elif mat == 'concrete':
            t1,t2 = random.randint(4,6), random.randint(5,8)
            t3,t4 = random.randint(30,40), random.randint(50,60)

        # t1,t2,t3,t4 = 4,9,14,24

        if time <= t1:
            return 'type1'
        elif t1 < time <= t2:
            return 'type2'
        elif t2 < time <= t3:
            return 'type3'
        elif t3 < time <= t4:
            return 'type4'
        else:
            return 'type5'

    def simulation(self, start_build, season, sim_time=180, seed=42, multi=False, njob=-1):
        self.wind_angle = self.seasons_attr[season][0]
        self.wind_speed = self.seasons_attr[season][1]
        season_firego = self._gdf[['id']].copy()
        season_firego[season] = season_firego['id'].apply(lambda x: self._get_radius_neighbors(self._gdf, x))
        self.season_near_id = season_firego
        self.status_process = None
        sim_res = pd.DataFrame()

        def single_sim(seed):
            building_status = {}
            building = self._gdf[['id']].copy()
            building['t'] = None
            building['status'] = 'type0'
            # initial state
            T = 0 

            first_bd = start_build
            building.loc[building['id']==first_bd, 't'] = 0
            building.loc[building['id']==first_bd, 'status'] = 'type1'

            status0 = building.drop(building[building['id'] == id].index)
            status1 = [first_bd]
            status2 = []
            status3 = []
            status4 = []

            # start simulation
            if multi:
                tt_range = range(1, sim_time+1)
            else:
                tt_range = tqdm.tqdm_notebook(range(1, sim_time+1))

            for tt in tt_range:
                T = tt
                # status of burning building
                status1_4 = status1+status2+status3+status4  # all burning buildings
                for firebld_id in status1_4:
                    time = T - int(building.loc[building['id']==firebld_id,'t'].iloc[0])
                    i_mat = self._gdf.loc[self._gdf['id']==firebld_id,'material'].iloc[0]
                    building.loc[building['id']==firebld_id,'status'] = self._status(time, i_mat, seed=seed)

                status0 = list(building[building['status']=='type0']['id'])
                status1 = list(building[building['status']=='type1']['id'])
                status2 = list(building[building['status']=='type2']['id'])
                status3 = list(building[building['status']=='type3']['id'])
                status4 = list(building[building['status']=='type4']['id'])

                status3_4 = status3+status4

                # domain mapping relationship Sij of all buildings on fire
                season_firego1 = season_firego[['id',season]].rename(columns={season:'nearid'})
                season_firego1 = season_firego1[season_firego1['id'].isin(status3_4)]

                if len(season_firego1) > 0 :
                    season_firego1['nearid'] = season_firego1['nearid'].apply(lambda x: eval(str(x))) 
                    season_firego1 = season_firego1.explode('nearid').dropna( axis=0, how='any')
                    season_firego1 = season_firego1[season_firego1['nearid'].isin(status0)]
                    if len(season_firego1) > 0 :
                        season_firego1['firepro'] = season_firego1.apply(lambda x: self._firepro(building,x['id'],x['nearid']),axis=1)
                        season_firego1 = season_firego1[['nearid','firepro']].groupby(['nearid']).max().reset_index()
                        
                        for row in season_firego1.itertuples():
                            if row.firepro > self.firepro_max:
                                building.loc[building['id']==row.nearid,'status'] = 'type1'
                                building.loc[building['id']==row.nearid,'t'] = T

                building_status[T] = building.copy()
            return building_status
        
        if multi:
            multi_results = Parallel(n_jobs=njob)(delayed(single_sim)(i) for i in tqdm.tqdm_notebook(range(100)))
            status_res = multi_results[seed+1]
            self.status_process = status_res

            for seed_id, status_res in enumerate(multi_results):
                building_num = {f'type{i}':[] for i in range(6)}
                for tt in range(1,sim_time+1):
                    temp = status_res[tt]
                    for i in range(6):
                        building_num[f'type{i}'].append(len(temp[temp['status']==f'type{i}']))
                building_num_df = pd.DataFrame(building_num, index=range(1,sim_time+1)).reset_index()
                sim_res = pd.concat([sim_res, building_num_df])

        else:
            status_res = single_sim(seed)
            self.status_process = status_res

            building_num = {f'type{i}':[] for i in range(6)}
            for tt in range(1, sim_time+1):
                temp = status_res[tt]
                for i in range(6):
                    building_num[f'type{i}'].append(len(temp[temp['status']==f'type{i}']))
            building_num_df = pd.DataFrame(building_num, index=range(1,sim_time+1)).reset_index()
            sim_res = pd.concat([sim_res, building_num_df])

        return sim_res, status_res[sim_time]

    def plot_process_curve(self, sim_res, ax=None):
        if ax is None:
            ax = plt.gca()
        for i in range(1,6):
            sns.lineplot(data=sim_res, x='index', y=f'type{i}', 
            color=self.mapping[f'type{i}'][1], 
            label=self.mapping[f'type{i}'][0], 
            ax=ax)
        ax.legend(frameon=False)
        ax.set_xlabel('time (min)')
        ax.set_ylabel('Number of buildings')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    def plot_spatial_map(self, plot_t, ax=None):
        temp = self.status_process[plot_t]
        _temp = pd.merge(self._gdf, temp, on='id', how='left')
        if ax is None:
            ax=plt.gca()
        self._gdf.plot(color='lightgrey', ax=ax)
        for i in range(1,6):
            if len(_temp[_temp['status']==f'type{i}']) != 0:
                _temp[_temp['status']==f'type{i}'].plot(ax=ax, color=self.mapping[f'type{i}'][1])
            else:
                pass        

    def fire_system(self, season, sim_time, seed=42, njob=-1):
        def single_sim2(id):
            try:
                _, last_status = self.simulation(id, season, sim_time=sim_time, seed=seed)
                threaten = len(last_status[last_status['status'] != 'type0'])
                fragile = last_status[last_status['status'] != 'type0']
                return threaten, fragile['id'].tolist()
            
            except GEOSException as e:
                print(f"Error with building id {id}: {e}")
                return 0, []

            
        multi_res = Parallel(n_jobs=njob)(delayed(single_sim2)(i) 
                                        for i in tqdm.notebook.tqdm(self._gdf['id'].tolist()))
        all_threaten = {i:j[0] for i,j in zip(self._gdf['id'].tolist(), multi_res)}
        all_fragile = [i[1] for i in multi_res]
        all_fragile = [i for j in all_fragile for i in j]
        all_fragile = dict(pd.DataFrame(all_fragile).value_counts(0))

        return all_fragile, all_threaten

if __name__ == '__main__':
    building_shp = gpd.read_file(r"E:\Research\firego\data\building_polygon.shp")
    building_shp = building_shp[building_shp.area >= 15].copy()
  
    road = gpd.read_file(r"E:\Research\firego\data\road_polygon.shp")
    river = gpd.read_file(r"E:\Research\firego\data\river_polygon.shp")
    stops = pd.concat([road[['geometry']], river[['geometry']]])

    # seasonal dominant wind direction and wind speed
    # wind direction is counted counterclockwise from the right side
    seasons_attr = {'spring':[270,4],'summer':[135,4],'autumn':[205,4],'winter':[115,2]}

    firego = FireGo(building_shp, seasons_attr, stops, firepro_max=0.3)

    # single building simulation 
    sim_result, last_status = firego.simulation(0, 'summer', sim_time=180, multi=False)
    fig,ax=plt.subplots(figsize=(5,5), dpi=150)
    firego.plot_process_curve(sim_result, ax=ax)

    # all buildings simulation
    all_fragile, all_threaten = firego.fire_system('summer', 180, seed=42)
