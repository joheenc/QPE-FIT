import os
smas = [100,200,400]
es = [0.01,0.1,0.3,0.5]
spins = [0.5,0.9]
Mbhs = [5,6,7]
windows_0 = '[[0,100000]]'
windows_1 = '[[0,100000],[300000,400000],[600000,700000],[900000,1000000]]'
windows_2 = '[[0,100000],[1100000,1200000],[2200000,2300000],[3300000,3400000]]'
windows_3 = '[[0,100000],[1100000,1200000],[2200000,2300000],[3300000,3400000],[4400000,4500000],[5500000,5600000],[6600000,6700000],[7700000,7800000],[8800000,8900000],[9900000,10000000]]'
windows = [windows_0, windows_1, windows_2, windows_3]

for sma in smas:
    for e in es:
        for spin in spins:
            for Mbh in Mbhs:
                for i, window in enumerate(windows):
                    if len(windows) > 3:
                        dt = 100
                    else:
                        dt = 10
                    preamble = f"#!/bin/bash\n#SBATCH --output=slurmfiles/sma={sma}_e={e}_a={spin}_Mbh={Mbh}_windows={i}.out\n#SBATCH --error=slurmfiles/sma={sma}_e={e}_a={spin}_Mbh={Mbh}_windows={i}.err\n#SBATCH --job-name=timing_sampler\n#SBATCH --partition=sched_mit_kburdge_r8\n#SBATCH --gres=gpu:1\n\n"
                    timing_file = f'timings_sma={sma}_e={e}_a={spin}_Mbh={Mbh}_windows={i}.dat'
                    window_file = f'windows_sma={sma}_e={e}_a={spin}_Mbh={Mbh}_windows={i}.dat'
                    outfile = f'h5files/sma={sma}_e={e}_a={spin}_Mbh={Mbh}_windows={i}.h5'
                    os.system(f'echo "{preamble}" > sample.sh')
                    cmd = f'python generate_timings.py {sma} {e} 60 {spin} {Mbh} {window} {timing_file} {window_file}'
                    os.system(f'echo "{cmd}" >> sample.sh')
                    cmd = f'python mcmc_fixedphase.py {outfile} {timing_file} {window_file} 20000 1000 {sma} {Mbh} {dt}'
                    os.system(f'echo "{cmd}" >> sample.sh')
                    os.system('sbatch -p sched_mit_kburdge_r8 --gres=gpu:1 sample.sh')
