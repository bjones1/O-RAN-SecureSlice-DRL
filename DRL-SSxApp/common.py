import random
import numpy as np


# ### **Get State**
#
# This function, get_state, simulates a state in a network slicing environment
# by randomly selecting data from three types of slices (eMBB, Medium, and
# UrLLC) and calculating the number of Physical Resource Blocks (PRBs) based on
# the selected data and a set of predefined rates (DL_BYTE_TO_PRB_RATES). The
# function also introduces a chance of one slice becoming "malicious" and
# increasing its DL bytes. The function returns the calculated PRBs for each
# slice.
def get_state(action_prbs, DL_BYTE_TO_PRB_RATES, malicious_chance):
    # Every time step there is a chance one slice becomes malicous (small) if a
    # slice is malicous the DL bytes will go way above the threashold
    RESET_VALUE = 10000
    MIN_CHANCE = 100
    if malicious_chance < MIN_CHANCE:
        malicious_chance = RESET_VALUE
    chance = random.randint(0,int(malicious_chance))


    if chance == malicious_chance:
        DL_BYTE_TO_PRB_RATES[random.randint(0,2)] *= 10
    # elif chance == 2000: DL_BYTE_TO_PRB_RATES\[1\] \*= 10 elif chance == 3000:
    # DL_BYTE_TO_PRB_RATES\[2\] \*= 10




    return [DL_BYTE_TO_PRB_RATES[0] * action_prbs[0],
            DL_BYTE_TO_PRB_RATES[1] * action_prbs[1],
            DL_BYTE_TO_PRB_RATES[2] * action_prbs[2]]



# ### **Perform Action**
#
# Simulates the outcome of taking an action in a given state.

# Actions 1-3: Adjust action_prbs based on state, incresaing prbs if the slice
# is operating within its SLA. Actions 4-6: Set specific action_prbs to 0, this
# secures slices operating over SLA and rewards agent for doing so.



def perform_action(action, state, i, action_prbs, DL_BYTES_THRESHOLD):


    reward = 0
    done = False
    next_state = np.copy(state)
    if sum(action_prbs) == 0:
        done = True
        return reward, done, action_prbs
    if reward == 27253551:
        done = True #max reward achieved.
        return reward, done, action_prbs
    if action < 3:
        for i, dl_bytes_value in enumerate(state):
            if action <=2:
                # if action_prbs\[action\] == 0: reward += 0

                action_prbs[action] += 5 #essentially we are mapping 5 more resource blocks to each slice the 3 UEs are in which is 5*6877 (DRL to PRB mapping). This is so we can speed up the increase of resources.
                if dl_bytes_value > DL_BYTES_THRESHOLD[i]:
                    action_prbs[i] -= 5
                    reward += 0
                else:
                    reward += dl_bytes_value
                    #reward /= len(state)
        


    else:
        action_prbs[action - 3] = 0
        if state[action - 3] > DL_BYTES_THRESHOLD[action - 3]:
            reward += max(state)
        else:
            reward += 0
    


    return reward, done, action_prbs

