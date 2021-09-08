'''ValueIteration.py
STUDENT STARTER TEMPLATE FOR ...
Value Iteration for Markov Decision Processes.
'''

# Edit the returned name to ensure you get credit for the assignment.
def student_name():
   return "Fang Yu" # For an autograder.

Vkplus1 = {}
Q_Values_Dict = {}

def one_step_of_VI(S, A, T, R, gamma, Vk):
   '''S is list of all the states defined for this MDP.
   A is a list of all the possible actions.
   T is a function representing the MDP's transition model.
   R is a function representing the MDP's reward function.
   gamma is the discount factor.
   The current value of each state s is accessible as Vk[s].
   '''

   '''Your code should fill the dictionaries Vkplus1 and Q_Values_dict
   with a new value for each state, and each q-state, and assign them
   to the state's and q-state's entries in the dictionaries, as in
       Vkplus1[s] = new_value
       Q_Values_Dict[(s, a)] = new_q_value

   Also determine delta_max, which we define to be the maximum
   amount that the absolute value of any state's value is changed
   during this iteration.
   '''
   global Q_Values_Dict
   global Vkplus1
   delta_max = 0
   for state in S:
      v_value = float('-inf')
      for action in A:
         q_value = 0
         for next_state in S:
            if next_state not in Vk:
               prev_value_next_state = 0
            else: 
               prev_value_next_state = Vk[next_state]
            q_value += T(state, action, next_state) * (R(state, action, next_state) + gamma * prev_value_next_state)
         Q_Values_Dict[(state, action)] = q_value
         if v_value < q_value:
            v_value = q_value
      
      if state in Vk:
         if delta_max < abs(v_value - Vk[state]): delta_max = abs(v_value - Vk[state])
      else:
         if delta_max < abs(v_value - 0): delta_max = abs(v_value - 0)
      
      Vkplus1[state] = v_value
      
   # global Q_Values_Dict
   
   return (Vkplus1, delta_max)
   # return (Vk, 0) # placeholder

def return_Q_values(S, A):
   '''Return the dictionary whose keys are (state, action) tuples,
   and whose values are floats representing the Q values from the
   most recent call to one_step_of_VI. This is the normal case, and
   the values of S and A passed in here can be ignored.
   However, if no such call has been made yet, use S and A to
   create the answer dictionary, and use 0.0 for all the values.
   '''
   global Q_Values_Dict
   if not Q_Values_Dict:
      for state in S:
         for action in A:
            Q_Values_Dict[(state, action)] = 0.0

   return Q_Values_Dict # placeholder

Policy = {}
def extract_policy(S, A):
   '''Return a dictionary mapping states to actions. Obtain the policy
   using the q-values most recently computed.  If none have yet been
   computed, call return_Q_values to initialize q-values, and then
   extract a policy.  Ties between actions having the same (s, a) value
   can be broken arbitrarily.
   '''
   global Policy
   global Q_Values_Dict
   Policy = {}
   if not Q_Values_Dict:
      Q_Values_Dict = return_Q_values(S, A)
   for pair in Q_Values_Dict:
      if pair[0] not in Policy:
         Policy[pair[0]] = pair[1]
      else:
         if Q_Values_Dict[(pair[0], Policy[pair[0]])] < Q_Values_Dict[pair]:
            Policy[pair[0]] = pair[1]
   # Add code here
   return Policy
 
def apply_policy(s):
   '''Return the action that your current best policy implies for state s.'''
   global Policy
   return Policy[s] # placeholder


