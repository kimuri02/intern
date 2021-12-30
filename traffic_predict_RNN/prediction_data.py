import traci
import pandas as pd
import xml.etree.ElementTree as ET

queue_7to8_list = []
queue_8to7_list = []
step_list = []
phase_list7 = []
phase_list8 = []
in_7_list = []
in_8_list = []
out_7_list = []
out_8_list = []
lane_id_list = []
veh_lane_id_list = []
veh_dict = {}

edge_7_in = []
edge_8_in = []

traffic_7_in = []
traffic_8_in = []


# edge_7_in = ['2to7', '6to7', '8to7', '12to7']
# edge_8_in = ['3to8', '7to8', '9to8', '13to8']

tree = ET.parse('hello_world.net.xml')
root = tree.getroot()

for i in root.iter('edge'):
    edge_id = i.get('id')
    if edge_id[0] != ':':
        if edge_id[-1] == '7':
            edge_7_in.append(edge_id)
        elif edge_id[-1] == '8':
            edge_8_in.append(edge_id)
# print(edge_8_in)


traci.start(["sumo-gui", "-c", "test.sumocfg", "--tripinfo-output", "./tripinfo.xml"])


while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()
    
    step = traci.simulation.getTime()

    queue_7to8 = traci.edge.getLastStepVehicleNumber('7to8')
    queue_8to7 = traci.edge.getLastStepVehicleNumber('8to7')

    if step % 7 == 0:
        queue_7to8_list.append(queue_7to8)
        queue_8to7_list.append(queue_8to7)

        phase_list7.append(traci.trafficlight.getPhase('gneJ0'))
        phase_list8.append(traci.trafficlight.getPhase('gneJ2'))

        sum_7_in = 0
        sum_8_in = 0

        for edge in edge_7_in:
            sum_7_in += traci.edge.getLastStepVehicleNumber(edge)            

        for edge in edge_8_in:
            sum_8_in += traci.edge.getLastStepVehicleNumber(edge)
            
        traffic_7_in.append(sum_7_in)
        traffic_8_in.append(sum_8_in)
        
traci.close()

column_name = ['step', '7to8', '8to7', 'phase_7', 'phase_8', '7in', '8in']
rows = []

for i in range(10,len(queue_8to7_list)-150):
    queue = [7*i, queue_7to8_list[i], queue_8to7_list[i], phase_list7[i], phase_list8[i], traffic_7_in[i], traffic_8_in[i]]
    rows.append(queue)

data_frame = pd.DataFrame(rows, columns = column_name)
data_frame.to_csv('7to8_data.csv', mode = 'w')