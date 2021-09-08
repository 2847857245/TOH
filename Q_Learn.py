'''Q_Learn.py
STUDENT STARTER TEMPLATE ...

Implement Q-Learning in this file by completing the implementations
of the functions whose stubs are present.
Add or change code wherever you see #*** ADD OR CHANGE CODE HERE ***

This is part of the UW Intro to AI Starter Code for Reinforcement Learning.

'''
import random
# Edit the returned name to ensure you get credit for the assignment.


def student_name():
    # *** ADD OR CHANGE CODE HERE ***
    return "Fang, Yu"  # For an autograder.


STATES = None
ACTIONS = None; UQV_callback = None; Q_VALUES = None
is_valid_goal_state = None
Terminal_state = None
USE_EXPLORATION_FUNCTION = None
INITIAL_STATE = None
Count = None


def setup(states, actions, q_vals_dict, update_q_value_callback,
          goal_test, terminal, use_exp_fn=False):
    '''This method is called by the GUI the first time a Q_Learning
    menu item is selected. It may be called again after the user has
    restarted from the File menu.
    Q_VALUES starts out with all Q-values at 0.0 and a separate key
    for each (s, a) pair.'''
    global STATES, ACTIONS, UQV_callback, Q_VALUES, is_valid_goal_state, Count
    global USE_EXPLORATION_FUNCTION, Terminal_state
    STATES = states
    ACTIONS = actions
    Q_VALUES = q_vals_dict
    UQV_callback = update_q_value_callback
    is_valid_goal_state = goal_test
    Terminal_state = terminal
    USE_EXPLORATION_FUNCTION = use_exp_fn
    if USE_EXPLORATION_FUNCTION:
        Count = {}
        for s in STATES:
            for a in ACTIONS[0:-1]:
                Count[(s,a)] = 1


PREVIOUS_STATE = None
LAST_ACTION = None


def set_starting_state(s):
    '''This is called by the GUI when a new episode starts.
    Do not change this function.'''
    global INITIAL_STATE, PREVIOUS_STATE
    print("In Q_Learn, setting the starting state to "+str(s))
    INITIAL_STATE = s
    PREVIOUS_STATE = s


ALPHA = 0.5
CUSTOM_ALPHA = False
EPSILON = 0.5
CUSTOM_EPSILON = False
GAMMA = 0.9


def set_learning_parameters(alpha, epsilon, gamma):
    ''' Called by the system. Do not change this function.'''
    global ALPHA, EPSILON, CUSTOM_ALPHA, CUSTOM_EPSILON, GAMMA
    ALPHA = alpha
    EPSILON = epsilon
    GAMMA = gamma
    if alpha < 0:
        CUSTOM_ALPHA = True
    else: CUSTOM_ALPHA = False
    if epsilon < 0:
        CUSTOM_EPSILON = True
    else: CUSTOM_EPSILON = False


def update_Q_value(previous_state, previous_action, new_value):
    '''Whenever your code changes a value in Q_VALUES, it should
    also call this method, so the changes can be reflected in the
    display.
    Do not change this function.'''
    UQV_callback(previous_state, previous_action, new_value)


def handle_transition(action, new_state, r):
    '''When the user drives the agent, the system will call this function,
    so that you can handle the learning that should take place on this
    transition.'''
    global PREVIOUS_STATE

# *** ADD OR CHANGE CODE HERE ***
    if ALPHA < 0:
        ALPHA = 0.3
    if EPSILON < 0:
        EPSILON = 0.3
    max_Q_Next = 0
    for act in ACTIONS:
        if (new_state, act) in Q_VALUES:
            if max_Q_Next < Q_VALUES[(new_state, act)]:
                max_Q_Next = Q_VALUES[(new_state, act)]
    
    new_qval = (1 - ALPHA) * Q_VALUES[(PREVIOUS_STATE, action)] + ALPHA * (r + GAMMA * max_Q_Next)

    Q_VALUES[(PREVIOUS_STATE, action)] = new_qval

    # You should call update_Q_value before returning.  E.g.,
    update_Q_value(PREVIOUS_STATE, action, new_qval)

    print("Transition to state: "+str(new_state) +
          "\n with reward "+str(r)+" is currently handled by your program.")
    PREVIOUS_STATE = new_state
    return  # Nothing needs to be returned.


def choose_next_action(s, r, terminated=False):
    '''When the GUI or engine calls this, the agent is now in state s,
    and it receives reward r.
    If terminated==True, it's the end of the episode, and this method
     can just return None after you have handled the transition.

    Use this information to update the q-value for the previous state
    and action pair.

    Then the agent needs to choose its action and return that.

    '''
    global INITIAL_STATE, PREVIOUS_STATE, LAST_ACTION, EPSILON, ALPHA, CUSTOM_ALPHA, CUSTOM_EPSILON, GAMMA, Count
    # Unless s is the initial state, compute a new q-value for the
    # previous state and action.
    
    if ALPHA < 0:
        ALPHA = 0.3
    if EPSILON < 0:
        EPSILON = 0.3

    if not (s == INITIAL_STATE):
        # Compute your update here.
        # if CUSTOM_ALPHA is True, manage the alpha values over time.
        # Otherwise go with the fixed value.
        new_qval = 0 
        max_Q_Next = 0

        if USE_EXPLORATION_FUNCTION:
            if LAST_ACTION != "Exit":
                Count[(PREVIOUS_STATE, LAST_ACTION)] += 1
            for act in ACTIONS:
                max_Q_Next = max(max_Q_Next, Q_VALUES[(s, act)] + 100 / Count[(s, act)])
        else:
            for act in ACTIONS:
                max_Q_Next = max(max_Q_Next, Q_VALUES[(s, act)])

        new_qval = (1 - ALPHA) * Q_VALUES[(PREVIOUS_STATE, LAST_ACTION)] + ALPHA * (r + GAMMA * max_Q_Next)

        # Save it in the dictionary of Q_VALUES:
        Q_VALUES[(PREVIOUS_STATE, LAST_ACTION)] = new_qval

        # Then let the Engine and GUI know about the new Q-value.
        update_Q_value(PREVIOUS_STATE, LAST_ACTION, new_qval)

    # Now select an action according to your Q-Learning criteria, such
    # as expected discounted future reward vs exploration.

    # If EPSILON > 0, or CUSTOM_EPSILON is True,
    # then use epsilon-greedy learning here.
    # In order to access q-values, simply get them from the dictionary, e.g.,
    # some_qval = Q_VALUES[(some_state, some_action)]

    some_action = ACTIONS[0]  # a placeholder, so some action is returned.

    max_qval = 0
    if is_valid_goal_state(s): # Exit if it's goal state
        if CUSTOM_EPSILON:
            EPSILON = EPSILON * 0.95
        if CUSTOM_ALPHA:
            ALPHA = ALPHA * 0.95
        some_action = "Exit"
    elif s == Terminal_state:
        some_action = None
    else:
        for action in ACTIONS[0: -1]:
            if max_qval < Q_VALUES[(s, action)]:
                some_action = action
                max_qval = Q_VALUES[(s, action)]

        # if USE_EXPLORATION_FUNCTION: # if use exploration function
        #     for action in ACTIONS[0: -1]:
        #         if max_qval < Q_VALUES[(s, action)] + 2 / Count[(s, action)]:
        #             some_action = action
        #             max_qval = Q_VALUES[(s, action)] + 2 / Count[(s, action)]

        if EPSILON > 0 and random.random() < EPSILON:
            some_action = random.choice(ACTIONS[0:-1])

    LAST_ACTION = some_action  # remember this for next time
    PREVIOUS_STATE = s  # "       "    "   "    "
    return some_action 


Policy = {}


def extract_policy(S, A):
    '''Return a dictionary mapping states to actions. Obtain the policy
    using the q-values most recently computed.
    Ties between actions having the same (s, a) value can be broken arbitrarily.
    Reminder: goal states should map to the Exit action, and no other states
    should map to the Exit action.
    '''
    global Policy
    Policy = {}
# *** ADD OR CHANGE CODE HERE ***
    for s in S:
        q_val = 0
        for act in A[0:-1]:
            if q_val < Q_VALUES[(s,act)]:
                best_action = act
                q_val = Q_VALUES[(s,act)]
        if is_valid_goal_state(s):
            best_action = "Exit"
        if s==Terminal_state:
            best_action = None
        if q_val == 0:
            best_action = ACTIONS[0]
        Policy[s] = best_action
    return Policy
