import simpy
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.stats
import copy
plt.style.use("seaborn")

QUEUE_MAX_CONTENT = float('inf')

ACTION_SPACES = 6  #[SPT,EDD,MST,ST,CR,WSPT]

class Product:
    def __init__(self, ID, j_type, arrival_time, process_time, due_dates):
        self.ID = ID
        self.type = int(j_type)
        self.arrival_time = arrival_time
        self.process_time = process_time
        self.due_dates = due_dates
        self.finish_time = None
        
class Source:
    def __init__(self, name, factory):
        self.name = name
        self.factory = factory
        self.env = factory.env
        #output: the number of jobs arrival
        self.output = 0
        self.inter_arrival = None
        
    def set_port(self, output_port):
        self.queue = output_port
        self.process = self.env.process(self.generate_product())
             
    def generate_product(self):
        for i in range(len(self.factory.JOB_DATA)):
            self.inter_arrival = self.factory.JOB_DATA[i][1] - self.factory.JOB_DATA[i-1][1] if i>=1 else self.factory.JOB_DATA[i][1]
            yield self.env.timeout(self.inter_arrival)
            self.output += 1
            product = Product(i, self.factory.JOB_DATA[i][0], self.factory.JOB_DATA[i][1], \
                              self.factory.JOB_DATA[i][2:2+self.factory.PROCESSORS_AVAILABLE], self.factory.JOB_DATA[i][2+self.factory.PROCESSORS_AVAILABLE])
            yield self.env.timeout(0)
            if self.queue.is_queue_full() == True:
                #print("{} : product {} ,type{} arrive".format(self.env.now, product.ID, product.type))
                self.queue.product_arrival(product)
                if product.ID + 1 < len(self.factory.JOB_DATA):
                    if product.arrival_time == self.factory.JOB_DATA[product.ID + 1][1]:
                        continue
                self.queue.check_direct_process()
                         
class Queue:
    def __init__(self, factory, max_content, name):
        self.name = name
        self.factory = factory
        self.env = factory.env
        self.queue = []
        self.max_content = max_content
        self.entity_type_now = np.zeros((self.factory.PROCESSORS_AVAILABLE,), dtype=np.int32)
        
    def set_port(self, output_port):
        self.processors = output_port
        
    def is_queue_full(self):
        assert len(self.queue) <= self.max_content
        if len(self.queue) == self.max_content:
            return False
        elif len(self.queue) < self.max_content:
            return True
        
    def check_direct_process(self):
        if len(self.queue) == 1:
            for i in range(len(self.processors)):
                if self.processors[i].is_free == True:
                    self.send_product(i)
                    break
                    
        if len(self.queue) > 1:
            for i in range(len(self.processors)):
                if self.processors[i].is_free == True:
                    self.factory.decision_point.succeed()
                    self.factory.decision_point = self.env.event()
                    self.env.process(self.wait_for_action())
            
    def get_product(self, i, p_t):
        if self.env.now not in [j[1] for j in self.factory.JOB_DATA]:
            if len(self.queue) > 1:
                self.factory.decision_point.succeed()
                self.factory.decision_point = self.env.event()
                self.env.process(self.wait_for_action())
            elif len(self.queue) == 1:
                self.send_product(i)
                
    def send_product(self, i):
        product = self.queue[0]
        self.entity_type_now[i] = product.type
        self.queue.pop(0)
        self.processors[i].process(product)
        
    def wait_for_action(self):
        yield self.factory.get_action
        for i in range(len(self.processors)):
            if self.processors[i].is_free == True and len(self.queue) > 0:
                #print('Product type in queue:',[i.type for i in self.queue])
                self.sort_queue(self.factory.dispatcher.action, i)
                self.send_product(i)
            
    def product_arrival(self, product):
        self.queue.append(product)
                  
    def sort_queue(self, rule_for_sorting, processor_id):
        if rule_for_sorting == 0:  #SPT
            self.queue.sort(key = lambda entity : entity.process_time[processor_id])
        elif rule_for_sorting == 1: #EDD
            self.queue.sort(key = lambda entity : entity.due_dates)
        elif rule_for_sorting == 2: #MST
            from_type = self.entity_type_now[processor_id] 
            if from_type != 0:
                self.queue.sort(key = lambda entity : self.factory.SET_UP_TIME[from_type - 1,entity.type - 1])
        elif rule_for_sorting == 3: #ST
            self.queue.sort(key = lambda entity : entity.due_dates - entity.process_time[processor_id])
        elif rule_for_sorting == 4: #CR
            self.queue.sort(key = lambda entity : entity.due_dates / entity.process_time[processor_id])
        elif rule_for_sorting == 5:  #WSPT
            self.queue.sort(key = lambda entity : entity.process_time[processor_id]/self.factory.WEIGHTS[entity.type -1])
        #print('action:{}, queue:{}'.format(rule_for_sorting, [p.ID for p in self.queue]))
            
class Processor:
    def __init__(self, factory, Processor_id, name):
        self.name = name
        self.factory = factory
        self.Processor_id = Processor_id
        self.is_free = True
        self.env = factory.env
        self.previous_product_type = 0
        self.processor_avail_time = 0
        self.last_calculate_st = 0
        
    def set_port(self, input_port, output_port):
        self.queue = input_port
        self.output = output_port
        
    def process(self,product):
        self.is_free = False
        #print("{} : product {} ,type{} start treating at processor{}".format(self.env.now, product.ID, product.type, self.Processor_id))
        self.env.process(self.processing(product))

    def processing(self, product):
        process_time = product.process_time[self.Processor_id] + self.factory.SET_UP_TIME[self.previous_product_type - 1][product.type - 1] \
        if self.previous_product_type != 0 else product.process_time[self.Processor_id]
        
        if process_time != product.process_time[self.Processor_id] or self.previous_product_type == 0:
            self.last_calculate_st = self.env.now
        
        self.processor_avail_time = process_time + self.env.now
        
        # update state
        self.factory.update_s_ps(product.ID, -1)
        self.factory.update_s_m3(self.Processor_id, self.previous_product_type, product.type, self.processor_avail_time)
        
        # compute_reward
        self.factory.compute_reward(self.env.now, process_time, product.ID)
        
        yield self.env.timeout(process_time)
        self.is_free = True
        #print("{} : product {} ,type{} finish treating at processor{}".format(self.env.now, product.ID, product.type, self.Processor_id))   
            
        if self.output == self.factory.sink:
            self.output.store(product)
        else:
            self.output.product_arrival(product)
            
        self.previous_product_type = product.type
        self.processor_avail_time = 0
        
        # update state
        self.factory.update_s_ps(product.ID, 1)
        self.factory.update_s_m3(self.Processor_id, self.previous_product_type, 0, self.processor_avail_time)

        # decision point
        self.queue.get_product(self.Processor_id, process_time)
        
class Sink:
    def __init__(self, factory):
        self.env = factory.env
        self.input = 0
        self.warehouse = []
        self.factory = factory
        self.number_of_late = np.zeros(len(self.factory.SET_UP_TIME))
          
    def store(self, product):
        self.input += 1 
        product.finish_time = self.env.now
        if product.finish_time > product.due_dates:
            self.number_of_late[product.type - 1] += 1
            
        self.factory.tardiness_set.append(max(product.finish_time - product.due_dates, 0))
        self.factory.cycle_time_set.append(product.finish_time - product.arrival_time)
        
        if self.input >= len(self.factory.JOB_DATA):
            self.factory.decision_point.succeed()
            self.factory.terminal.succeed()
            
        self.warehouse.append(product)
    
    def update(self):
        self.warehouse.clear()
        
class Dispatcher:
    def __init__(self, factory):
        self.action = None
        self.factory = factory
    
    def assign_dispatching_rule(self, action):
        assert action in np.arange(ACTION_SPACES)
        self.factory.get_action.succeed()
        self.factory.get_action = self.factory.env.event()
        self.action = action

class Factory:
    def __init__(self, n):
        #columns=[ job_type, arrival_time, process_time, due_dates]
        self.JOB_DATA = pd.read_csv('Experiment_data/job_data'+str(n)+'.csv').to_numpy()
        #determine setup_time by job_type
        self.SET_UP_TIME = pd.read_csv('Experiment_data/setup_times'+str(n)+'.csv').to_numpy()
        self.WEIGHTS = np.ones(len(self.JOB_DATA))
        self.PROCESSORS_AVAILABLE = 4
        #WEIGHTS[1] = 10
        
    def build(self):  
        self.env = simpy.Environment()
        self.processor_1_available = self.PROCESSORS_AVAILABLE
        self.queue_1 = Queue(self, QUEUE_MAX_CONTENT, 'queue_1')
        self.processors_1 = [] 
        self.source = Source('source_1', self)
        self.sink = Sink(self)
        self.dispatcher = Dispatcher(self)
        
        self.source.set_port(self.queue_1)
        self.queue_1.set_port(self.processors_1)
        self.append_processor(
            self.processors_1, self.processor_1_available, 'processor_1', self.queue_1, self.sink
        )
        
        self.cycle_time_set = []
        self.tardiness_set = []
        
        #making action event
        self.decision_point = self.env.event()
        
        #get action event
        self.get_action = self.env.event()
        
        #terminal event
        self.terminal   = self.env.event()
        
        self.observation = self.get_initial_state()
        self.reward = 0
        
    def get_state(self):
        self.update_s_t(self.env.now)
        self.update_s_lcst()
        return copy.deepcopy(self.observation)
        
    def get_reward(self):
        return self.reward
        
    def append_processor(self, processors, num, name, input_port, output_port):
        for i in range(num):
            processor = Processor(self, i, name)
            processor.set_port(input_port, output_port)
            processors.append(processor)
    
    def reset(self):
        self.build()
        self.env.run(until = self.decision_point)
        initial_state = self.get_state()
        return initial_state
    
    def step(self, action):
        self.dispatcher.assign_dispatching_rule(action)
        self.env.run(until = self.decision_point)
        state = self.get_state()
        reward = self.get_reward()
        done = self.terminal.triggered
        
        weights = np.array(self.WEIGHTS, dtype = np.float32)
        weights = weights / np.sum(weights)
        
        #mean weighted tardiness, makespan, cycle time, tardiness
        info = [np.sum(weights * self.sink.number_of_late), self.env.now, np.mean(self.cycle_time_set), np.mean(self.tardiness_set)]
        
        self.reset_reward()
        return state, reward, done, info
        
    #state method
    def get_initial_state(self):
        matrix_1 = np.zeros((len(self.JOB_DATA), 5+self.PROCESSORS_AVAILABLE), dtype = np.float32)
        matrix_2 = np.array(self.SET_UP_TIME, dtype = np.float32)
        matrix_3 = np.zeros((self.PROCESSORS_AVAILABLE, 4), dtype = np.float32)
        for i in range(len(matrix_1)):
            matrix_1[i][:len(self.JOB_DATA[i])] = self.JOB_DATA[i]

        return [matrix_1, matrix_2, matrix_3]
    
    def update_s_ps(self, job_id, process_state):
        self.observation[0][job_id,3+self.PROCESSORS_AVAILABLE] = process_state
        
    def update_s_t(self, time):
        self.observation[0][:,4+self.PROCESSORS_AVAILABLE] = time
        
    def update_s_m3(self, machine_id, previous_j_type, now_j_type, mat):
        self.observation[2][machine_id,:-1] = [previous_j_type, now_j_type, mat]
    
    def update_s_lcst(self):
        for processor in self.processors_1:
            self.observation[2][processor.Processor_id,-1] = self.env.now - processor.last_calculate_st
            
    #reward method
    def compute_reward(self, start_process_t, process_t, job_id):
        weights = np.array(self.WEIGHTS, dtype = np.float32)
        weights = weights / np.sum(weights)
        Latest_start_process_t = self.JOB_DATA[job_id][2+self.PROCESSORS_AVAILABLE] - process_t
        max_delay = Latest_start_process_t - self.JOB_DATA[job_id][1]
        reward = (Latest_start_process_t - start_process_t)/max_delay if max_delay > 0 else -100
        weighted_reward = weights[int(self.JOB_DATA[job_id][0] - 1)] * reward if reward >= 0 else weights[int(self.JOB_DATA[job_id][0] - 1)] * -100

        self.reward += weighted_reward

    def reset_reward(self):
        self.reward = 0      