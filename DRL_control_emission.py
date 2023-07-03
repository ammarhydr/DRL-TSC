#         Traffic Signal Controller with RL models on SUMO
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Ammar Haydari
# Date: 2023
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TSC enviroment interacts with SUMO including vehicle level emission statistics




import re
import traci
import random
import numpy as np
import math

class DRL_Agent:
    
    def __init__(self, tsc_data, sim):
        self.tsc_data = tsc_data
        self.sim = sim
        self.episode=0
        
    def reset_env(self):
        
        self.queue_length = 0
        self.co2 = 0
        self.waiting_time = 0
        self.action = 0
        self.prev_action = 0
        self.green_time=0
        self.timer=0
        self.y_timer=0
        self.reward=0
        self.const=0
        self.n_reward=0
        self.reward1=0
        self.agent_act=True
        self.state_data = {'state':(), 'next_state':()} 
        self.all_reward = []
        self.queue_length_node = 0
        self.peak_check=False
        self.peak=0
        self.peak_time = []
        self.max_green=0
        self.max_green_cons=0
        self.num_veh=[]
        self.total_emission=[]
        self.speed_mean=[]
        self.speed_var=[]
        self.stopping_vehs=[]
        self.emission = 0
        self.emission_pred = 0
        self.emission_pred_all = []
        self.episode += 1
        self.delay = 0
        self.total_vehicle = 0
        
    def getState(self, tsc_data, node):

        s_state=[]
        max_distance = 200
        veh_length = 5
        max_veh_number=max_distance/veh_length
        
        
        lanes=tsc_data[node]['phase_data']['lanes']
        position = np.zeros((len(lanes), 1))
        velocity = np.zeros((len(lanes), 1))
        i=-1                   
        for lane in lanes:
            i+=1                   
            vehicles = traci.lane.getLastStepVehicleIDs(lane)
            for v in vehicles:
                junctionPosition = traci.junction.getPosition(node)
                vehicleposition = traci.vehicle.getPosition(v)
                dist = math.hypot(junctionPosition[0] - vehicleposition[0], junctionPosition[1] - vehicleposition[1])
                if dist <= max_distance:
                    position[i] += 1/max_veh_number
                    velocity[i] += traci.vehicle.getSpeed(v) / traci.lane.getMaxSpeed(lane)                        
        position = position.reshape(1, len(lanes))
        velocity = velocity.reshape(1, len(lanes))
        n_state=np.concatenate((position,velocity), axis=0)
        s_state.append(n_state)
        
        state_local = np.array(s_state[0])
        

        if len(tsc_data)>1:    
            for nnode in tsc_data[node]['phase_data']['neighbors']:
                lanes=tsc_data[nnode]['phase_data']['lanes']
                position = np.zeros((len(lanes), 1))
                velocity = np.zeros((len(lanes), 1))
                i=-1                   
                for lane in lanes:
                    i+=1                   
                    vehicles = traci.lane.getLastStepVehicleIDs(lane)
                    for v in vehicles:
                        junctionPosition = traci.junction.getPosition(nnode)
                        vehicleposition = traci.vehicle.getPosition(v)
                        dist = math.hypot(junctionPosition[0] - vehicleposition[0], junctionPosition[1] - vehicleposition[1])
                        if dist <= max_distance:
                            position[i] += 1/max_veh_number
                            velocity[i] += traci.vehicle.getSpeed(v) / traci.lane.getMaxSpeed(lane)                        
                position = position.reshape(1, len(lanes))
                velocity = velocity.reshape(1, len(lanes))
                n_state=np.concatenate((position,velocity), axis=0)
                s_state.append(n_state)
            state=np.concatenate(tuple(s_state), axis=1)
        else:
            state=np.array(s_state[0])
        return state_local.reshape(-1,1).transpose(), state.reshape(-1,1).transpose()
        
    def env_step(self, agent_node):
        agent, node = agent_node[0], agent_node[1]
        phase_data = self.tsc_data[node]['phase_data']
        if self.timer== 0 and self.y_timer== 0:
            if self.agent_act == False:
                self.green_phase=phase_data['green_phases'][self.action]
                self.max_green = 10 * (len(re.findall("G", self.green_phase))/2)
                self.max_green_cons = 15 * (len(re.findall("G", self.green_phase))/2)
                self.timer = phase_data['durations'][phase_data['phases'].index(self.green_phase)]
                traci.trafficlight.setRedYellowGreenState(node, self.green_phase)
                # reward1[indx] = sim.stopping_vehs_norm(node)
            else:
                _, self.state_data['state'] = self.getState(self.tsc_data, node)
                self.prev_action = self.action
                self.action = agent.act(self.state_data['state'])
                # print(self.action)
                if self.action != self.prev_action:
                    #reset green timer for max_green checking
                    if self.green_time > self.max_green: 
                        self.peak_time.append(self.green_time - self.max_green)
                    self.green_phase = phase_data['green_phases'][self.action]
                    self.green_time = phase_data['durations'][phase_data['phases'].index(self.green_phase)]
                    
                    #add yellow transition
                    current_phase = phase_data['green_phases'][self.prev_action]
                    yellow_phase = phase_data['phases'][phase_data['phases'].index(current_phase)+1]
                    self.y_timer = phase_data['durations'][phase_data['phases'].index(current_phase)+1]
                    traci.trafficlight.setRedYellowGreenState(node, yellow_phase)
                    self.agent_act = False
                else:
                    #check max green duration
                    self.green_phase = phase_data['green_phases'][self.action]                            
                    self.green_time += phase_data['durations'][phase_data['phases'].index(self.green_phase)]
                    self.timer = phase_data['durations'][phase_data['phases'].index(self.green_phase)]
                    traci.trafficlight.setRedYellowGreenState(node, self.green_phase)
                    # reward1[indx] = - sim.stopping_vehs_norm(node)
                    if self.green_time >= self.max_green: 
                        # print('Peak violation occured')
                        self.peak_check=True
                        # if self.green_time >= self.max_green_cons:                          
                        #     self.peak+=1
                        if self.green_time >= 5*self.max_green:
                            # forcefully select a different action and reset green_timer
                            self.timer=0
                            while self.action == self.prev_action:
                                self.green_phase = random.choice(phase_data['green_phases'])
                                self.action = phase_data['green_phases'].index(self.green_phase)                        
                            self.green_time = phase_data['durations'][phase_data['phases'].index(self.green_phase)]
                            #add yellow transition
                            current_phase = phase_data['green_phases'][self.prev_action]
                            yellow_phase = phase_data['phases'][phase_data['phases'].index(current_phase)+1]
                            self.y_timer = phase_data['durations'][phase_data['phases'].index(current_phase)+1]
                            traci.trafficlight.setRedYellowGreenState(node, yellow_phase)
                            self.agent_act = False   
                            
        elif self.timer == 0 and self.y_timer != 0:
            self.y_timer=self.y_timer-1
        elif self.timer != 0 and self.y_timer == 0:
            self.timer=self.timer-1
            if self.timer == 0: 
                self.agent_act = True 
                _, self.state_data['next_state'] = self.getState(self.tsc_data, node)
                # reward[indx] = reward[indx] - reward1[indx] #- sim.delay(node)
                # reward1[indx] = 0
                # if self.peak_check==True:
                #     # if node=='gneJ1':
                #     # self.reward = - self.sim.stopping_vehs(phase_data['roads']) - 0.06*(self.green_time)
                #     self.reward = - self.sim.stopping_vehs(phase_data['roads'])# - 0.1*(self.green_time-self.max_green)
                #     self.const = (self.green_time-self.max_green)
                #     # self.const = -100
                #     # print(self.green_phase)
                #     # else:
                #         # self.reward = - self.sim.stopping_vehs(phase_data['roads'])
                #     self.peak_check = False
                # else:
                    # self.reward = - self.sim.stopping_vehs(phase_data['roads'])
                    # self.const  = 0
                co2_av, total_veh, _ = self.sim.emission_co2_pervehicle(phase_data['roads'])
                self.emission += co2_av*total_veh
                # self.emission_pred += 1800*total_veh + 4000 
                self.emission_pred += 1250*total_veh + 4000 
                self.emission_pred_all.append(self.emission_pred)
                m_speed, v_speed=self.sim.speed_stats(phase_data['roads'])
                # print(f"Mean speed: {m_speed}, Speed Variance: {v_speed}")

                ###################
                # wait_time=self.sim.overall_waitingtime(node) 
                stopping_veh = self.sim.stopping_vehs(phase_data['roads'])
                # self.reward = - (0.25*stopping_veh + wait_time)  
                ###################
                self.delay += self.sim.delay(node)
                # nl=(len(re.findall("G", self.green_phase))/2)
                self.reward = - (self.delay + 15*stopping_veh)/60
                # print(f"#SV: {stopping_veh}, #delay: {self.delay/60}")
                self.delay = 0
                ###################

                if self.emission > self.emission_pred:
                    self.const = 0.005*(self.emission - self.emission_pred)
                    self.peak+=1
                    # print(self.const)
                    # For every end of action instance
                    # self.reward = - self.sim.stopping_vehs(phase_data['roads']) - 0.02 * (self.emission - self.emission_pred)  
                    if self.episode>50:
                        self.reward -= self.const

                    # print(f"Reward: {self.reward}, Constraint: {self.const}")

                #reset the emission statistics
                self.emission = 0
                self.emission_pred = 0

                # av_acceleration=self.sim.acceleration(phase_data['roads'])
                # print(f"#veh: {total_veh}, av speed: {speed_av}, av co2: {co2_av}, av accel: {av_acceleration}")
                self.num_veh.append(total_veh)
                self.total_emission.append(co2_av*total_veh)
                self.speed_mean.append(m_speed)
                self.speed_var.append(v_speed)
                self.stopping_vehs.append(self.sim.stopping_vehs(phase_data['roads']))
                # reward[indx] = reward1[indx] - sim.overall_waitingtime(node)
                # print(self.reward)
                self.all_reward.append(self.reward)
                agent.train_model(self.state_data['state'], self.action, self.reward, self.state_data['next_state'], False)
                    
        self.queue_length += self.sim.stopping_vehs(phase_data['roads'])
        co2_av, total_veh, speed_av = self.sim.emission_co2_pervehicle(phase_data['roads'])
        self.co2 += co2_av*total_veh
        self.total_vehicle += total_veh
        self.delay += self.sim.delay(node)
        
        # self.emission += co2_av*total_veh
        # self.emission_pred += 2000*total_veh + 7000
        