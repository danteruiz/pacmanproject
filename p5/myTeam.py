# myTeam.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util
from game import Directions
from game import Actions
import game
from util import nearestPoint
from util import manhattanDistance

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.

  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DefensiveAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''
    self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
    self.initializeUniformly(gameState)

    self.firstMove = True
    self.prob_attack = 0.8
    self.prob_scaredFlee = 0.8
    self.index_ = None







  def getPositionDistribution(self, gameState):
    """
    Returns a distribution over successor positions of the ghost from the given gameState.

    You must first place the ghost in the gameState, using setGhostPosition below.
    """
    ghostPosition = gameState.getAgentState(self.index_).getPosition() # The position you set
    actionDist = self.getDistribution(gameState)
    dist = util.Counter()
    for action, prob in actionDist.items():
      successorPosition = game.Actions.getSuccessor(ghostPosition, action)
      dist[successorPosition] = prob
    return dist

  def setGhostPosition(self, gameState, ghostPosition):
    """
    Sets the position of the ghost for this inference module to the specified
    position in the supplied gameState.
    """
    conf = game.Configuration(ghostPosition, game.Directions.STOP)
    gameState.data.agentStates[self.index_] = game.AgentState(conf, False)
    return gameState



  def observeState(self, gameState):
    distances = gameState.getAgentDistances()
    #if len(distances) >= self.index: # Check for missing observations
    obs = distances[ self.index_ ]
    self.observe(obs, gameState)


  def initializeUniformly(self, gameState):
    "Begin with a uniform distribution over ghost positions."
    self.opponents_beliefs = {}
    for index in self.getOpponents(gameState):
      print "Index: ", index
      beliefs = util.Counter()
      for p in self.legalPositions:
        beliefs[p] = 1.0
        beliefs.normalize()
      self.opponents_beliefs[index] = beliefs



  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    for index in self.getOpponents(gameState):
      print 'this is the index', index
      if not self.firstMove: self.elapseTime(gameState)
      self.firstMove = False
      self.index_ = index
      self.observeState(gameState)
    opponents_beliefs_list = self.opponents_beliefs.items()
    opponents_beliefs_vals_only = []
    for agentIndex, beliefVal in opponents_beliefs_list:
      opponents_beliefs_vals_only.append(beliefVal)
    self.displayDistributionsOverPositions(opponents_beliefs_vals_only)

    '''
     we need to change this next two lines

    '''

    actions = gameState.getLegalActions(self.index)
    return random.choice(actions)



  def observe(self, observation, gameState):
    current = self.opponents_beliefs[self.index_]
    noisyDistance = observation
    emissionModel = getObservationDistribution(noisyDistance)
    pacmanPosition = gameState.getAgentState(self.index).getPosition()

    # Replace this code with a correct observation update
    allPossible = util.Counter()
    for p in self.legalPositions:
      trueDistance = util.manhattanDistance(p, pacmanPosition)
      if emissionModel[trueDistance] > 0: allPossible[p] = emissionModel[trueDistance] * current[p]
    allPossible.normalize()

    self.opponents_beliefs[self.index_] = allPossible




  def elapseTime(self, gameState):
    current = self.opponents_beliefs[self.index_]
    counter = util.Counter()

    for p in self.legalPositions:
         newPosDist = self.getPositionDistribution(self.setGhostPosition(gameState, p))

         for nextPos in newPosDist:
             counter[nextPos] += newPosDist[nextPos] * current[p]



    self.opponents_beliefs[self.index_] = counter

  def getBeliefDistribution(self):
    return self.opponents_beliefs[self.index_]


  def getDistribution( self, state ):
    # Read variables from state
    ghostState = state.getAgentState(self.index_ )
    legalActions = state.getLegalActions( self.index_)
    pos = ghostState.getPosition();
    isScared = ghostState.scaredTimer > 0

    speed = 1
    if isScared: speed = 0.5

    actionVectors = [Actions.directionToVector( a, speed ) for a in legalActions]
    newPositions = [( pos[0]+a[0], pos[1]+a[1] ) for a in actionVectors]
    pacmanPosition = state.getAgentState(self.index).getPosition()

    # Select best actions given the state
    distancesToPacman = [manhattanDistance( pos, pacmanPosition ) for pos in newPositions]
    if isScared:
      bestScore = max( distancesToPacman )
      bestProb = self.prob_scaredFlee
    else:
      bestScore = min( distancesToPacman )
      bestProb = self.prob_attack
    bestActions = [action for action, distance in zip( legalActions, distancesToPacman ) if distance == bestScore]

    # Construct distribution
    dist = util.Counter()
    for a in bestActions: dist[a] = bestProb / len(bestActions)
    for a in legalActions: dist[a] += ( 1-bestProb ) / len(legalActions)
    dist.normalize()
    return dist











class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)

    # Compute distance to the nearest food
    foodList = self.getFood(successor).asList()
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1}

class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}











def chooseAction(self, gameState):
    """
    First computes the most likely position of each ghost that
    has not yet been captured, then chooses an action that brings
    Pacman closer to the closest ghost (in maze distance!).

    To find the maze distance between any two positions, use:
    self.distancer.getDistance(pos1, pos2)

    To find the successor position of a position after an action:
    successorPosition = Actions.getSuccessor(position, action)

    livingGhostPositionDistributions, defined below, is a list of
    util.Counter objects equal to the position belief distributions
    for each of the ghosts that are still alive.  It is defined based
    on (these are implementation details about which you need not be
    concerned):

      1) gameState.getLivingGhosts(), a list of booleans, one for each
         agent, indicating whether or not the agent is alive.  Note
         that pacman is always agent 0, so the ghosts are agents 1,
         onwards (just as before).

      2) self.ghostBeliefs, the list of belief distributions for each
         of the ghosts (including ghosts that are not alive).  The
         indices into this list should be 1 less than indices into the
         gameState.getLivingGhosts() list.

    You may remove Directions.STOP from the list of available actions.
    """
    pacmanPosition = gameState.getPacmanPosition()
    legal = [a for a in gameState.getLegalPacmanActions() if a != Directions.STOP]
    livingGhosts = gameState.getLivingGhosts()
    livingGhostPositionDistributions = [beliefs for i,beliefs
                                        in enumerate(self.ghostBeliefs)
                                        if livingGhosts[i+1]]
    possible_location = []
    max_prob = -2
    location = (0,0)

    for dict in livingGhostPositionDistributions:
        for position in dict:
            prob = dict[position]

            if prob > max_prob:
                max_prob = prob
                location = position

        possible_location.append(location)


    max_distance = 100000
    max_position = None

    for i in possible_location:
        dist = self.distancer.getDistance(pacmanPosition, i)

        if dist < max_distance:
            max_distance = dist
            max_position = i

    closest = 1000000
    best_action = None
    for action in legal:
        successorPosition = Actions.getSuccessor(pacmanPosition, action)

        dist = self.distancer.getDistance(successorPosition, max_position)

        if dist < closest:
            closest = dist
            best_action = action

    return best_action



SONAR_NOISE_RANGE = 15 # Must be odd
SONAR_MAX = (SONAR_NOISE_RANGE - 1)/2
SONAR_NOISE_VALUES = [i - SONAR_MAX for i in range(SONAR_NOISE_RANGE)]
SONAR_DENOMINATOR = 2 ** SONAR_MAX  + 2 ** (SONAR_MAX + 1) - 2.0
SONAR_NOISE_PROBS = [2 ** (SONAR_MAX-abs(v)) / SONAR_DENOMINATOR  for v in SONAR_NOISE_VALUES]

observationDistributions = {}
def getObservationDistribution(noisyDistance):
  """
  Returns the factor P( noisyDistance | TrueDistances ), the likelihood of the provided noisyDistance
  conditioned upon all the possible true distances that could have generated it.
  """
  global observationDistributions
  if noisyDistance not in observationDistributions:
    distribution = util.Counter()
    for error , prob in zip(SONAR_NOISE_VALUES, SONAR_NOISE_PROBS):
      distribution[max(1, noisyDistance - error)] += prob
    observationDistributions[noisyDistance] = distribution
  return observationDistributions[noisyDistance]