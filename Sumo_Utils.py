#         Traffic Signal Controller with RL models on SUMO
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Ammar Haydari
# Date: 2023
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Essential function for collecting traffic flow statistics from SUMO using traci


import sys 
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
    
import random    
import sumolib
import traci
import math
import numpy as np

from sumolib import checkBinary

class SimData:
    def __init__(self, netfile):
        
        self.tgd = 6
        self.tgl = 4
        self.ty  = 3
        self.max_distance=300
        self.netfile  = netfile
        self.net = sumolib.net.readNet(self.netfile)
        self.tscs = traci.trafficlight.getIDList()


    def get_tsc_data(self, neighbors, nodes):
        
        all_nns = sum([neighbors[nn] for nn in neighbors], [])
        all_nodes = list(set(all_nns+nodes))
        tsc_data = {name:{} for name in all_nodes}
        
        for tsc in tsc_data:
            
            phase_info=traci.trafficlight.getAllProgramLogics(tsc)[0]    
            green_phases = []
            all_phases = []
            durations = []
            for phase in phase_info.phases:
                if ('g' in phase.state or 'G' in phase.state) and 'y' not in phase.state:
                    green_phases.append(phase.state)
                all_phases.append(phase.state)
                if (phase.duration) >= 18:
                    durations.append(self.tgd)
                elif (phase.duration) <= 18 and (phase.duration) >= 5:
                    durations.append(self.tgl)
                elif (phase.duration) == 3:
                    durations.append(self.ty)
                
            all_lanes= traci.trafficlight.getControlledLanes(tsc)
            roads=[]
            lanes = []
            for lane in all_lanes:
                if lane not in lanes:
                    lanes.append(lane)
                road=traci.lane.getEdgeID(lane)
                if road not in roads:
                    roads.append(road)

            if len(tsc_data)>1:    
                nnodes_all=self.net.getNode(tsc).getNeighboringNodes()
                nnodeIDs=[nnode.getID() for nnode in nnodes_all]
                nnodes=[nodeID for nodeID in nnodeIDs if nodeID in self.tscs]
                
                phase_data={'phases':all_phases, 'durations': durations, 'green_phases':green_phases, 'roads':roads, 'lanes':lanes, 'neighbors': nnodes}
                tsc_data[tsc]['phase_data'] = phase_data
            else:
                phase_data={'phases':all_phases, 'durations': durations, 'green_phases':green_phases, 'roads':roads, 'lanes':lanes}
                tsc_data[tsc]['phase_data'] = phase_data

            # all_links=traci.trafficlight.getControlledLinks(tsc)
            # incoming_lanes=[]
            # outgoing_lanes=[]
            # for link in all_links:
            #     if link[0][0] not in incoming_lanes:
            #         incoming_lanes.append(link[0][0])
            #     if link[0][1] not in outgoing_lanes:
            #         outgoing_lanes.append(link[0][1]) 
            
            # if len(tsc_data)>1:    
            #     phase_data={'phases':all_phases, 'durations': durations, 'green_phases':green_phases, 'roads':roads,'lanes':lanes, 'incoming_lanes':incoming_lanes, 'outgoing_lanes':outgoing_lanes, 'neighbors': neighbors[tsc]}
            #     tsc_data[tsc]['phase_data'] = phase_data
            # else:
            #     phase_data={'phases':all_phases, 'durations': durations, 'green_phases':green_phases, 'roads':roads,'lanes':lanes, 'incoming_lanes':incoming_lanes, 'outgoing_lanes':outgoing_lanes}
            #     tsc_data[tsc]['phase_data'] = phase_data
            
        return tsc_data

    def select_nodes(self):
        
        # all_nns = sum([neighbors[nn] for nn in neighbors], [])
        # all_nodes = list(set(all_nns+nodes))
        nodes=list(traci.trafficlight.getIDList())        
        selected_nodes=[]
        for tsc in nodes:
            
            phase_info=traci.trafficlight.getAllProgramLogics(tsc)[0]    
            green_phases = []
            for phase in phase_info.phases:
                if ('g' in phase.state or 'G' in phase.state) and 'y' not in phase.state:
                    green_phases.append(phase.state)
            
            if len(green_phases)>2:
                selected_nodes.append(tsc)  
        return selected_nodes
            
            
    def get_neighbors(self, nodes):
        # tscs = traci.trafficlight.getIDList()
        neighbors={ name:{} for name in nodes }
        for node in nodes:
            nnodes=self.net.getNode(node).getNeighboringNodes()
            nodeIDs=[nnode.getID() for nnode in nnodes]
            node_tscs=[nodeID for nodeID in nodeIDs if nodeID in self.tscs]
            
            neighbors[node]=node_tscs
        return neighbors
    
    def overall_traveltime(self, node):
        travel_time=0
        lanes = tuple(set(traci.trafficlight.getControlledLanes(node)))
        for lane in lanes:
            travel_time += (traci.lane.getTraveltime(lane)/60)
        return travel_time
    
    def overall_waitingtime(self, node):
        wait_time=0
        lanes = tuple(set(traci.trafficlight.getControlledLanes(node)))
        for lane in lanes:
            vehicles=traci.lane.getLastStepVehicleIDs(lane)
            if vehicles!=0:
                for v in vehicles:
                    junctionPosition = traci.junction.getPosition(node)
                    vehicleposition = traci.vehicle.getPosition(v)
                    dist = math.hypot(junctionPosition[0] - vehicleposition[0], junctionPosition[1] - vehicleposition[1])
                    if dist <= self.max_distance:
                        wait_time += traci.vehicle.getWaitingTime(v)       
            # wait_time += (traci.lane.getWaitingTime(lane)) #seconds
            # wait_time += (traci.lane.getWaitingTime(lane)/60) #minutes
        return wait_time
    
    def delay(self,node):
        delay=0
        lanes = tuple(set(traci.trafficlight.getControlledLanes(node)))
        for lane in lanes:
            vehicles=traci.lane.getLastStepVehicleIDs(lane)
            if vehicles!=0:
                for v in vehicles:
                    junctionPosition = traci.junction.getPosition(node)
                    vehicleposition = traci.vehicle.getPosition(v)
                    dist = math.hypot(junctionPosition[0] - vehicleposition[0], junctionPosition[1] - vehicleposition[1])
                    if dist <= self.max_distance:
                        delay += 1 - traci.vehicle.getSpeed(v) / traci.lane.getMaxSpeed(lane) 
                # delay += sum([1 - traci.vehicle.getSpeed(vehicle) / traci.lane.getMaxSpeed(lane) for vehicle in vehicles])
                # delay += sum([(traci.lane.getMaxSpeed(lane) /traci.vehicle.getSpeed(vehicle) - 1) for vehicle in vehicles if traci.vehicle.getSpeed(vehicle)!=0])
            # delay += 1 - traci.lane.getLastStepMeanSpeed(lane) / traci.lane.getMaxSpeed(lane)
        return delay
    
    def stopping_vehs(self, roads):
        wait=0
        for road in roads:
            wait += traci.edge.getLastStepHaltingNumber(road) 
        return wait
    
    def stopping_vehs_norm(self, node):
        # wait_time=0
        stops = 0
        vehicles = 0
      
        lanes = tuple(set(traci.trafficlight.getControlledLanes(node)))
        for lane in lanes:
            # wait_time += (traci.lane.getWaitingTime(lane)) #seconds
            stops += traci.lane.getLastStepHaltingNumber(lane)
            vehicles += traci.lane.getLastStepVehicleNumber(lane)
            
        if vehicles != 0:
            return stops/vehicles
        else:
            return 0
    
    # def stopping_vehs_norm(self, roads):
    #     norm_num_vehs = 0
    #     for road in roads:
    #         if traci.edge.getLastStepVehicleNumber(road) != 0:
    #             norm_num_vehs += (traci.edge.getLastStepHaltingNumber(road) / traci.edge.getLastStepVehicleNumber(road))
    #     return norm_num_vehs

    def overall_fuel(self, node):
        fuel=0
        lanes = tuple(set(traci.trafficlight.getControlledLanes(node)))
        for lane in lanes:
            Vehicles=traci.lane.getLastStepVehicleNumber(lane)
            if Vehicles!=0:
                fuel += (traci.lane.getFuelConsumption(lane)/(Vehicles*1000))
        return fuel
    
    def overall_co2(self, node):
        co2=0
        lanes = tuple(set(traci.trafficlight.getControlledLanes(node)))
        for lane in lanes:
            Vehicles=traci.lane.getLastStepVehicleNumber(lane)
            if Vehicles!=0:
                co2 += (traci.lane.getCO2Emission(lane)/(Vehicles*1000))
        return co2

    def emission_fuel_pervehicle(self,roads):
        fuel=0
        total_veh=0
        for road in roads:
            fuel += traci.edge.getFuelConsumption(road)
            total_veh += traci.edge.getLastStepVehicleNumber(road)
        return fuel/total_veh, total_veh   
    
    def emission_co2_pervehicle(self,roads):
        co2=0
        total_veh=0
        av_speed=0
        for road in roads:
            co2 += traci.edge.getCO2Emission(road)
            total_veh += traci.edge.getLastStepVehicleNumber(road)
            av_speed += traci.edge.getLastStepMeanSpeed(road)
        if total_veh > 0:
            return round(co2/total_veh, 2), total_veh, round(av_speed/len(roads), 2)   
        else:
            return 0, 0, round(av_speed/len(roads), 2)  
        
    def emission_fuel(self,roads):
        fuel=0
        for road in roads:
            fuel += traci.edge.getFuelConsumption(road) 
        return fuel

    def emission_co2(self,roads):
        co2=0
        for road in roads:
            co2 += traci.edge.getCO2Emission(road) 
        return co2
    
    def vehs_in_lane(self,lanes):
        vehicles=0
        for lane in lanes:
            vehicles += traci.lane.getLastStepVehicleNumber(lane) 
        return vehicles

    def speed_stats(self, roads):
        speeds=[]
        for road in roads:
            vehicles=traci.edge.getLastStepVehicleIDs(road)
            if len(vehicles)!=0:
                speed=[traci.vehicle.getSpeed(vehicle) for vehicle in vehicles]
                speeds+=speed
        mean_speed = np.mean(speeds) if len(speeds)>0 else 0
        var_speed = np.var(speeds) if len(speeds)>0 else 0
        return mean_speed, var_speed

    def acceleration(self, roads):
        acceleration=0
        counter=0
        for road in roads:
            vehicles=traci.edge.getLastStepVehicleIDs(road)
            if len(vehicles)>0:
                acceleration+= sum([traci.vehicle.getAcceleration(vehicle) for vehicle in vehicles])
                counter+=len(vehicles)
                # if traci.vehicle.getAcceleration(vehicle)>=0:
                    # acceleration += traci.vehicle.getAcceleration(vehicle) 
        av_acceleration = acceleration/counter if not counter==0 else 0
        return round(av_acceleration,2)

    def get_mp_lanes(self, netfile, tsc_data):
        net = sumolib.net.readNet(netfile)
        nodes_info = net.getNodes()
        mp_lanes={ tsc:{} for tsc in tsc_data }
        for tsc in tsc_data:
            try:
                node_data={}
                for node in nodes_info:
                    if tsc==node.getID():
                        node_data['tlsindex'] = { conn.getTLLinkIndex():str(conn.getFromLane().getID()) for conn in node.getConnections()}
                
                green_phases=tsc_data[tsc]['phase_data']['green_phases']
                phase_lanes = {phase:[] for phase in green_phases}
                for phase in green_phases:
                    green_lanes = set()
                    red_lanes = set()
                    for s in range(len(phase)):
                        if phase[s] == 'g' or phase[s] == 'G':
                            green_lanes.add(node_data['tlsindex'][s])
                        elif phase[s] == 'r':
                            red_lanes.add(node_data['tlsindex'][s])    
                
                    pure_green = [l for l in green_lanes if l not in red_lanes]
                    if len(pure_green) == 0:
                        phase_lanes[phase] = list(set(green_lanes))
                    else:
                        phase_lanes[phase] = list(set(pure_green))
                
                all_lanes= traci.trafficlight.getControlledLanes(tsc)
                lanes_data = [net.getLane(lane) for lane in all_lanes]
                lane_data = {lane:{} for lane in all_lanes}
                for lane in lanes_data:
                    lane_id = lane.getID()
                    lane_data[ lane_id ]['outgoing'] = {}
                    for conn in lane.getOutgoing():
                        out_id = str(conn.getToLane().getID())
                        lane_data[ lane_id ]['outgoing'][out_id] = {'dir':str(conn.getDirection()), 'index':conn.getTLLinkIndex()}
                
                max_pressure_lanes = {}
                for g in green_phases:
                    inc_lanes = set()
                    out_lanes = set()
                    for l in phase_lanes[g]:
                        inc_lanes.add(l)
                        for ol in lane_data[l]['outgoing']:
                            out_lanes.add(ol)
                    max_pressure_lanes[g] = {'inc':inc_lanes, 'out':out_lanes}
                
                mp_lanes[tsc] = max_pressure_lanes
            except:
                print(tsc)

        return mp_lanes
    
    def num_vehicle(self, node):
        lanes=traci.trafficlight.getControlledLanes(node)
        vehicles=0
        for lane in lanes:
            vehicles+=traci.lane.getLastStepVehicleNumber(lane)
        return vehicles


    def max_pressure(self, green_phases, max_pressure_lanes, node):
        phase_pressure = {}
        no_vehicle_phases = []
        max_distance=200
        #compute pressure for all green movements
        for g in green_phases:
            inc_lanes = max_pressure_lanes[g]['inc']
            out_lanes = max_pressure_lanes[g]['out']
            
            i=-1
            position = np.zeros((len(inc_lanes), 1))
            for lane in inc_lanes:
                i+=1                   
                vehicles = traci.lane.getLastStepVehicleIDs(lane)
                for v in vehicles:
                    junctionPosition = traci.junction.getPosition(node)
                    vehicleposition = traci.vehicle.getPosition(v)
                    dist = math.hypot(junctionPosition[0] - vehicleposition[0], junctionPosition[1] - vehicleposition[1])
                    if dist <= max_distance:
                        position[i] += 1
            inc_pressure = sum(position)

            i=-1
            position = np.zeros((len(out_lanes), 1))
            for lane in out_lanes:
                i+=1                   
                vehicles = traci.lane.getLastStepVehicleIDs(lane)
                for v in vehicles:
                    junctionPosition = traci.junction.getPosition(node)
                    vehicleposition = traci.vehicle.getPosition(v)
                    dist = math.hypot(junctionPosition[0] - vehicleposition[0], junctionPosition[1] - vehicleposition[1])
                    if dist <= max_distance:
                        position[i] += 1
            out_pressure = sum(position)
            
            #pressure is defined as the number of vehicles in a lane
            phase_pressure[g] = inc_pressure - out_pressure
            if inc_pressure == 0 and out_pressure == 0:
                no_vehicle_phases.append(g)

        ###if no vehicles randomly select a phase
        if len(no_vehicle_phases) == len(green_phases):
            return random.choice(green_phases)
        else:
            phase_pressure = [ (p, phase_pressure[p]) for p in phase_pressure]
            phase_pressure = sorted(phase_pressure, key=lambda p:p[1], reverse=True)
            phase_pressure = [ p for p in phase_pressure if p[1] == phase_pressure[0][1] ]
            return random.choice(phase_pressure)[0]
if __name__ == '__main__':
    cfgfile="/media/ahaydari/2TB_extra/Rl_files/TSCs/mysumo_adv/Network/SF_downtown/SF_downtown.sumocfg"
    # actuated_cfgfile='Network/SF_downtown/SF_downtown_actuated.sumocfg'
    netfile='/media/ahaydari/2TB_extra/Rl_files/TSCs/mysumo_adv/Network/SF_downtown/SF_downtown.net.xml'
    
    
    sumoBinary = checkBinary('sumo')
    traci.start([sumoBinary, "-c", cfgfile, '--start', '--collision.check-junctions', '--collision.action=remove','-W','--quit-on-end'])
    sim=SimData(netfile)
    # nodes = ['65306793','65306795','65306797','65317030','65334510','65363153', '65317042', '65317045','65306789','65308413']
    # nodes = ['65306793','65306797','65317030','65334510', '65317045','65306789']
    # nodes = ['65306793','65306797','65317030','65334510', '65317045','65306789']
    nodes = list(traci.trafficlight.getIDList())
    
    neighbors = sim.get_neighbors(nodes)
    tsc_data  = sim.get_tsc_data(neighbors, nodes)
    mp_lanes = sim.get_mp_lanes(netfile, tsc_data)
    traci.close(wait=False) 