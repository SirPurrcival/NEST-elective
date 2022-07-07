

# Import libraries
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import nest
import itertools

## 

# Create classes
class Network():
    def __init__(self, num_neurons, rho, eps, g, eta, J, neuron_params, n_rec_ex, n_rec_in, rec_start, rec_stop):
        self.num_neurons = num_neurons
        self.num_ex = int((1 - rho) * num_neurons)  # number of excitatory neurons
        self.num_in = int(rho * num_neurons)        # number of inhibitory neurons
        self.c_ex = int(eps * self.num_ex)          # number of excitatory connections
        self.c_in = int(eps * self.num_in)          # number of inhibitory connections
        self.J_ex = J                               # excitatory weight
        self.J_in = -g*J                            # inhibitory weight
        self.n_rec_ex = n_rec_ex                    # number of recorded excitatory neurons
        self.n_rec_in = n_rec_in                    # number of recorded inhibitory neurons
        self.rec_start = rec_start
        self.rec_stop = rec_stop
        self.neuron_params = neuron_params          # neuron params
        self.ext_rate = (self.neuron_params['V_th']   # the external rate needs to be adapted to provide enough input (Brunel 2000)
                         / (J * self.c_ex * self.neuron_params['tau_m'])
                         * eta * 1000. * self.c_ex)

    def create(self):

        # Create the network

        # First create the neurons - the excitatory and inhibitory populations
        ## Your code here
        self.neurons = nest.Create('iaf_psc_delta', self.num_neurons, params=self.neuron_params)
        self.neurons_ex = self.neurons[:self.num_ex]
        self.neurons_in = self.neurons[self.num_ex:]

        # Then create the external poisson spike generator
        ## Your code here
        
        self.p_noise = nest.Create("poisson_generator")
        self.p_noise.rate = self.ext_rate
        #print("Tself.ext_rate" + str(self.ext_rate))

        # Then create spike detectors
        # (disclaimer: dependening on NEST version, the device might be 'spike_detector' or 'spike_recorder'
        ## Your code here
        self.spike_recorder_ex = nest.Create("spike_recorder", 1)
        self.spike_recorder_in = nest.Create("spike_recorder", 1)
        # Next we connect the excitatory and inhibitory neurons to each other, choose a delay of 1.5 ms
        nest.Connect(self.neurons_ex, self.neurons,
             conn_spec={'rule': 'fixed_indegree', 'indegree': self.c_ex},
             syn_spec={'weight': self.J_ex, 'delay': 1.5})
        # Now also connect the inhibitory neurons to the other neurons
        ## Your code here
        nest.Connect(self.neurons_in, self.neurons,
                     conn_spec={'rule': 'fixed_indegree', 'indegree': self.c_in},
                     syn_spec={'weight': self.J_in, 'delay': 1.5})

        # Then we connect the external drive to the neurons with weight J_ex
        ## Your code here
        nest.Connect(self.p_noise, self.neurons,
                     syn_spec={'weight': self.J_ex, 'delay': 1.5})

        # Then we connect the the neurons to the spike detectors
        # Note: You can use slicing for nest node collections as well
        ## Your code here
        print(len(self.neurons_in[:self.n_rec_in]), self.n_rec_in)
        nest.Connect(self.neurons_ex[:self.n_rec_ex], self.spike_recorder_ex)
        nest.Connect(self.neurons_in[:self.n_rec_in], self.spike_recorder_in)

    def simulate(self, t_sim):
        # Simulate the network with specified
        nest.Simulate(t_sim)

    def get_data(self):
        # Define lists to store spike trains in
        self.spikes_ex = []
        self.spikes_in = []
        
        self.spike_times_ex = nest.GetStatus(self.spike_recorder_ex)
        self.spike_times_in = nest.GetStatus(self.spike_recorder_in)
        
        self.tmp_in = np.empty(self.n_rec_in, dtype=object)
        self.tmp_in[...] = [[] for _ in range(self.tmp_in.shape[0])]
        self.tmp_ex = np.empty(self.n_rec_ex, dtype=object)
        self.tmp_ex[...] = [[] for _ in range(self.tmp_ex.shape[0])]
        # There are several ways in which you can obtain the data recorded by the spikerecorders
        # One example is given below.
        # You can get the recorded quantities from the spike recorder with nest.GetStatus
        # You may loop over the entries of the GetStatus return
        # you might want to sort the spike times, they are not by default
        ## Your code here
        
        for i in range(len(self.spike_times_ex[0]['events']['senders'])):
            idx = self.spike_times_ex[0]['events']['senders'][i] - min(self.spike_times_ex[0]['events']['senders'])
            self.tmp_ex[idx].append(self.spike_times_ex[0]['events']['times'][i])
            
        for i in range(len(self.spike_times_in[0]['events']['senders'])):
            idx = self.spike_times_in[0]['events']['senders'][i] - min(self.spike_times_in[0]['events']['senders'])
            self.tmp_in[idx].append(self.spike_times_in[0]['events']['times'][i])
            
        
        #for item in self.spike_times_ex:
        #    self.spikes_ex.append(np.sort(item['events']['times']))
        # 
        #for item in self.spike_times_in:
        #    self.spikes_in.append(np.sort(item['events']['times'])) #['times']
        
        
        # hint: another option would be to obtain both the times and the senders (neurons).
        # This way you obtain information about which neuron spiked at which time.
        # e.g. senders = nest.GetStatus(self.spikes_recorder, 'events')[0]['senders']
        #      times   = nest.GetStatus(self.spikes_recorder, 'events')[0]['times']
        # Try to practice with the nest.GetStatus command.
        
        return self.tmp_ex, self.tmp_in

# Helper functions
def raster(spikes_ex, spikes_in, rec_start, rec_stop, figsize=(9, 5)):

    spikes_ex_total = list(itertools.chain(*spikes_ex))
    spikes_in_total = list(itertools.chain(*spikes_in))
    spikes_total = spikes_ex_total + spikes_in_total

    n_rec_ex = len(spikes_ex)
    n_rec_in = len(spikes_in)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(5, 1)

    ax1 = fig.add_subplot(gs[:4,0])
    ax2 = fig.add_subplot(gs[4,0])

    ax1.set_xlim([rec_start, rec_stop])
    ax2.set_xlim([rec_start, rec_stop])

    ax1.set_ylabel('Neuron ID')

    ax2.set_ylabel('Rate [Hz]')
    ax2.set_xlabel('Time [ms]')

    for i in range(n_rec_in):
        ax1.plot(spikes_in[i],
                 i*np.ones(len(spikes_in[i])),
                 linestyle='',
                 marker='o',
                 color='r',
                 markersize=2)
    for i in range(n_rec_ex):
        ax1.plot(spikes_ex[i],
                 (i + n_rec_in)*np.ones(len(spikes_ex[i])),
                 linestyle='',
                 marker='o',
                 color='b',
                 markersize=2)

    ax2 = ax2.hist(spikes_ex_total,
                   range=(rec_start,rec_stop),
                   bins=int(rec_stop - rec_start))

    plt.tight_layout(pad=1)

    plt.savefig('raster.png')

def rate(spikes_ex, spikes_in, rec_start, rec_stop):
    spikes_ex_total = list(itertools.chain(*spikes_ex))
    spikes_in_total = list(itertools.chain(*spikes_in))
    spikes_total = spikes_ex_total + spikes_in_total

    n_rec_ex = len(spikes_ex)
    n_rec_in = len(spikes_in)

    time_diff = (rec_stop - rec_start)/1000.
    average_firing_rate = (len(spikes_total)
                           / time_diff
                           /(n_rec_ex + n_rec_in))
    print(f'Average firing rate: {average_firing_rate} Hz')

neuron_params = {"C_m":     1.0,
                 "tau_m":   20.,
                 "t_ref":   2.0,
                 "E_L":     0.0,
                 "V_reset": 0.0,
                 "V_m":     0.0,
                 "V_th":    20.
                 }

params = {
    'num_neurons': 8000,                # number of neurons in network
    'rho':  0.2,                        # fraction of inhibitory neurons
    'eps':  0.2,                        # probability to establish a connections
    'g':    5,                          # excitation-inhibition balance
    'eta':  3.5,                          # relative external rate
    'J':    0.1,                        # postsynaptic amplitude in mV
    'neuron_params': neuron_params,     # single neuron parameters
    'n_rec_ex':  600,                   # excitatory neurons to be recorded from
    'n_rec_in':  150,                   # inhibitory neurons to be recorded from
    'rec_start': 600.,                  # start point for recording spike trains
    'rec_stop':  800.                   # end points for recording spike trains
    }

nest.ResetKernel()
nest.SetKernelStatus({'local_num_threads': 4})  # Adapt if necessary

nest.print_time = True
#nest.overwrite_files = True

network = Network(**params)
network.create()
network.simulate(1000)
test = network.get_data()
rate(test[0], test[1], params.get('rec_start'), params.get('rec_stop'))

#nu_th = theta / (J * CE * tauMem)
#nu_ex = eta * nu_th

#theta = V_th

nu_th = neuron_params['V_th'] / (params['J'] * network.c_ex * neuron_params['tau_m'])
nu_ex = params['eta'] * nu_th

ratio = nu_ex/nu_th
print(f"Parameters for this network: Nu_ex/Nu_thr = {ratio}, g = {params['g']}, Delay = 1.5ms")

raster(test[0], test[1], params.get('rec_start'), params.get('rec_stop'))
#nest.raster_plot.from_device(network.spike_recorder_ex)