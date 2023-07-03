#         Traffic Signal Controller with RL models on SUMO
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Ammar Haydari
# Date: 2023
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This python script controls traffic signals on any network using Reinforcement Learning modles and SUMO.

# The author has coded and run the scripts mainly on Spyder!

import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
    
from sumolib import checkBinary
import traci

# import matplotlib.pyplot as plt
from Sumo_Utils import SimData

from DRL_control_emission import DRL_Agent
from SAC_Agent import SACAgent

import time
import numpy as np

# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'qt')


        
        
if __name__ == '__main__':

    single=False
    fixed=True
    actuated=True
    maxpressure=False    
    filename="Results_SAC_06302023.txt"
    actormodel='SAC_actor_model_'
    criticmodel='SAC_critic_model_'
    

    # route.generate_routefile_mixed_nocross()
    cfgfile="Networks/cross_mixed_cross3ltl.sumocfg.sumocfg"
    actuated_cfgfile="Networks/actuated_mixed_cross3ltl.sumocfg"

    cfgfile="SUMO_cfg_xml/network1.sumocfg"
    netfile='SUMO_cfg_xml/network1.net.xml'  


    sumoBinary = checkBinary('sumo-gui')
    traci.start([sumoBinary, "-c", cfgfile, '--start', '--collision.check-junctions', '--collision.action=remove','-W','--quit-on-end'])
    sim=SimData(netfile)
    nodes = list(traci.trafficlight.getIDList())
    neighbors=sim.get_neighbors(nodes)
    tsc_data  = sim.get_tsc_data(neighbors, nodes)
    mp_lanes = sim.get_mp_lanes(netfile, tsc_data)
    traci.close(wait=False) 

    # parameters
    # batch_size = 64
    episodes = 50
    peak = []
    peak_all = []
    co2_all = []
    total_veh_all=[]
    
    lr = 0.00005 #multi
    # lr = 0.0001 #single


    av_reward = { node:[] for node in nodes }
    av_queue = { node:[] for node in nodes }
    av_co2 = { node:[] for node in nodes }
    av_peak_time = { node:[] for node in nodes }
    
    agents={ node:{} for node in nodes }
    drl_env={ node:{} for node in nodes }
    mp_env={ node:{} for node in nodes }
    for node in nodes:
        n_actions=len(tsc_data[node]['phase_data']['green_phases'])
        if len(tsc_data)>1:
            n_lanes=len(tsc_data[node]['phase_data']['lanes'])
            for neighbor in neighbors[node]:
                n_lanes+= len(tsc_data[neighbor]['phase_data']['lanes'])
        else:
            n_lanes=len(tsc_data[node]['phase_data']['lanes'])

        agents[node]=SACAgent(n_actions, n_lanes*2, lr)
        drl_env[node] = DRL_Agent(tsc_data, sim)
        
        drl_env[node].reset_env()
        
        # mp_env[node] = Max_Pressure(tsc_data, mp_lanes, sim)
        # mp_env[node].reset_env()
    
    # i=0
    # for agent in agents:
    #     try:
    #         agent.load(actormodel + str(i) + '.pth', criticmodel + str(i) + '.pth')
    #         i+=1
    #     except:
    #         print('No models found')    
    
    
    
    # Fixed Time Traffic Controller
    step = 0
    co2_f=0 
    fuel_f = 0
    queue_length_fixed = 0
    # mean_speed_f = []
    # mean_acceleration_f = []
    if fixed==True:        
        traci.start([sumoBinary, "-c", cfgfile, '--start', "--no-step-log", "true", "-W",'--quit-on-end']) 
        while traci.simulation.getMinExpectedNumber() > 0 and step < 10000:
            step+=1
            traci.simulationStep()
            for node in nodes:
                roads=tsc_data[node]['phase_data']['roads']
                queue_length_fixed += sim.stopping_vehs(roads)
                co2_f+=sim.emission_co2(roads)
                fuel_f+=sim.emission_fuel(roads)
                # if e==episodes-1:
                #     mean_speed_f.append(sim.mean_speed(roads))
                #     mean_acceleration_f.append(sim.acceleration(roads))
        traci.close(wait=False)

    # Actuated Traffic Controller
    step = 0
    queue_length_actuated = 0
    co2_a=0 
    fuel_a = 0
    # mean_speed_a = []
    # mean_acceleration_a = []
    if actuated==True:
        traci.start([sumoBinary, "-c", actuated_cfgfile, '--start', "--no-step-log", "true", "-W",'--quit-on-end']) 
        # traci.start([sumoBinary, "-c", actuated_cfgfile, '--start','--collision.check-junctions', '--collision.action=remove','-W','--quit-on-end']) 
        while traci.simulation.getMinExpectedNumber() > 0 and step < 10000:
            step+=1
            traci.simulationStep()
            for node in nodes:
                roads=tsc_data[node]['phase_data']['roads']
                queue_length_actuated += sim.stopping_vehs(roads)
                co2_a+=sim.emission_co2(roads)
                fuel_a+=sim.emission_fuel(roads)
                # if e==episodes-1:
                #     mean_speed_a.append(sim.mean_speed(roads))
                #     mean_acceleration_a.append(sim.acceleration(roads))
        traci.close(wait=False)  


    for e in range(episodes):
        
        start = time.time()
        
        #start DRL Traffic Controller
        step=0
        traci.start([sumoBinary, "-c", cfgfile, '--start', "--no-step-log", "true", "-W",'--quit-on-end']) 
        # traci.start([sumoBinary, "-c", cfgfile, '--start', '--collision.check-junctions', '--collision.action=remove','-W','--quit-on-end'])                 
        while traci.simulation.getMinExpectedNumber() > 20 and step < 10000:
            step+=1
            print(step) if step%1000==0 else None
            traci.simulationStep()
            for node in nodes:
                drl_env[node].env_step([agents[node], node])
        traci.close(wait=False)
        queue_length=sum([drl_env[node].queue_length for node in nodes])
        co2=sum([drl_env[node].co2 for node in nodes])
        total_veh=sum([drl_env[node].total_vehicle for node in nodes])
        peak.append([drl_env[node].peak for node in nodes])
        peak_all.append(sum(peak[-1]))
        co2_all.append(co2)
        total_veh_all.append(total_veh)
        print('total peak violation is: ',sum(peak[-1]))
        
        node=nodes[0]
        d_n=drl_env[node]
        # num_veh=d_n.num_veh        
        num_veh_sort=np.sort(d_n.num_veh).reshape(-1, 1)
        indx=np.argsort(d_n.num_veh)
        total_emission_sort=np.array(d_n.total_emission)[indx].reshape(-1, 1)
        speed_mean_sort=np.array(d_n.speed_mean)[indx].reshape(-1, 1)
        speed_var_sort=np.array(d_n.speed_var)[indx].reshape(-1, 1)
        stopping_vehs_sort=np.array(d_n.stopping_vehs)[indx].reshape(-1, 1)
        emission_pred_sl=np.array(d_n.emission_pred_all)[indx].reshape(-1, 1)
        av_emission=total_emission_sort/num_veh_sort
        
        
        
        #collect final RL parameters at the end of episode and reset everything
        for node in nodes:
            d_n=drl_env[node]
            if d_n.green_time >= d_n.max_green_cons: 
                d_n.peak_time.append(d_n.green_time - d_n.max_green_cons)
            av_peak_time[node].append(np.mean(d_n.peak_time))
            av_reward[node].append(np.mean(d_n.all_reward))
            av_queue[node].append(d_n.queue_length)
            av_co2[node].append(d_n.co2/d_n.total_vehicle)
            agents[node].train_model(d_n.state_data['state'], d_n.action, d_n.reward, d_n.state_data['next_state'], True)
            d_n.reset_env()
            agents[node].first_update=True


        # print('episode - ' + str(e) + ' SAC - ' + str(queue_length)+ ' fixed - ' + str(queue_length_fixed)+ ' actuated - ' + str(queue_length_actuated)+ ' maxpressure - ' + str(queue_length_mp))
        # print('episode - ' + str(e) + ' SAC - co2 ' + str(co2/total_veh))
        with open(filename, "a") as f:
            f.write("Simulation {}: SAC {} fixed {} actuated {} maxpressure {} \n".format(str(e), queue_length, queue_length_fixed, 0, 0))
        
        # print('Max reward:',max_reward, 'min reward:', min_reward )
        # print('Waiting time for episode - ' + str(e) + ' SAC - ' + str(waiting_time))        

        
for node, agent in agents:
    agent.save(actormodel + node +'.pth', criticmodel + node+'.pth')
sys.stdout.flush()
