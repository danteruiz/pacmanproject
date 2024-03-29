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
               first = 'OffInfRefAgent', second = 'InferenceReflexAgent'):
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

class InferenceReflexAgent(CaptureAgent):

  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
    self.initializeUniformly(gameState)
    self.firstMove = True
    self.prob_attack = 0.8
    self.prob_scaredFlee = 0.8
    self.index_ = None
    self.enemy_states = {}

  def getPositionDistribution(self, gameState):
    ghostPosition = gameState.getAgentState(self.index_).getPosition() # The position you set
    actionDist = self.getDistribution(gameState)
    dist = util.Counter()
    for action, prob in actionDist.items():
      successorPosition = game.Actions.getSuccessor(ghostPosition, action)
      dist[successorPosition] = prob
    return dist

  def setGhostPosition(self, gameState, ghostPosition):
    conf = game.Configuration(ghostPosition, game.Directions.STOP)
    gameState.data.agentStates[self.index_] = game.AgentState(conf, False)
    return gameState

  def observeState(self, gameState):
    distances = gameState.getAgentDistances()
    obs = distances[ self.index_ ]
    self.observe(obs, gameState)

  def initializeUniformly(self, gameState):
    self.opponents_beliefs = {}
    for index in self.getOpponents(gameState):
      beliefs = util.Counter()
      for p in self.legalPositions:
        beliefs[p] = 1.0
        beliefs.normalize()
      self.opponents_beliefs[index] = beliefs

  def chooseAction(self, gameState):
    for index in self.getOpponents(gameState):
      agent = gameState.getAgentState(index)
      self.enemy_states[index] = agent
      #print agent
      if not self.firstMove: self.elapseTime(gameState)
      self.firstMove = False
      self.index_ = index
      self.observeState(gameState)
    opponents_beliefs_list = self.opponents_beliefs.items()
    opponents_beliefs_vals_only = []
    for agentIndex, beliefVal in opponents_beliefs_list:
      opponents_beliefs_vals_only.append(beliefVal)
    self.displayDistributionsOverPositions(opponents_beliefs_vals_only)
    actions = gameState.getLegalActions(self.index)
    values = [self.evaluate(gameState, a) for a in actions]
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    return random.choice(bestActions)

  def observe(self, observation, gameState):
    current = self.opponents_beliefs[self.index_]
    noisyDistance = observation
    emissionModel = getObservationDistribution(noisyDistance)
    pacmanPosition = gameState.getAgentState(self.index).getPosition()
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
    ghostState = state.getAgentState(self.index_ )
    legalActions = state.getLegalActions( self.index_)
    pos = ghostState.getPosition();
    isScared = ghostState.scaredTimer > 0
    speed = 1
    if isScared: speed = 0.5
    actionVectors = [Actions.directionToVector( a, speed ) for a in legalActions]
    newPositions = [( pos[0]+a[0], pos[1]+a[1] ) for a in actionVectors]
    pacmanPosition = state.getAgentState(self.index).getPosition()
    distancesToPacman = [manhattanDistance( pos, pacmanPosition ) for pos in newPositions]
    if isScared:
      bestScore = max( distancesToPacman )
      bestProb = self.prob_scaredFlee
    else:
      bestScore = min( distancesToPacman )
      bestProb = self.prob_attack
    bestActions = [action for action, distance in zip( legalActions, distancesToPacman ) if distance == bestScore]
    dist = util.Counter()
    for a in bestActions: dist[a] = bestProb / len(bestActions)
    for a in legalActions: dist[a] += ( 1-bestProb ) / len(legalActions)
    dist.normalize()
    return dist

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
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['invaderDistance'] = 0

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    invaders = 0
    min_dist = 1000000
    for agent_index in self.getOpponents(successor):
       candidates = []
       for position, prob in self.opponents_beliefs[agent_index].items():
         candidates.append((prob, position))
       maxCandidate = max(candidates)
       agent = self.enemy_states[agent_index]
       if not agent.isPacman: continue
       invaders += 1


       if agent.getPosition() != None:
           features['invaderDistance'] = self.getMazeDistance(myPos,agent.getPosition())
           if(manhattanDistance(myPos,agent.getPosition()) <= 0.7):
             self.initializeUniformly(gameState)

       else:
           features['invaderDistance'] = self.getMazeDistance(myPos,maxCandidate[1])



       #dists = [self.getMazeDistance(myPos, self.getBeliefsPosition(a)) for a in invaders]
       #features['invaderDistance'] = min(dists)

    features['numInvaders'] = invaders

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}







#Offensive Agent
class OffInfRefAgent(CaptureAgent):

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
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    features['enemyDistance'] = 0
    features['reverse'] = 0

    if self.red:
      enemies = gameState.getBlueTeamIndices()
    else:
      enemies = gameState.getRedTeamIndices()


    myPosition = successor.getAgentState(self.index).getPosition()

    if successor.getAgentPosition(enemies[1]):
      defensePosition = successor.getAgentPosition(enemies[1])
      enemyDefenseAgentDist = manhattanDistance(myPosition, defensePosition)
    else:
      enemyDefenseAgentDist = successor.getAgentDistances()[enemies[1]]

    if enemyDefenseAgentDist <= 3:
      #print "Too close!"
      if successor.getAgentState(enemies[1]).scaredTimer > 0:
        features['enemyDistance'] = -(enemyDefenseAgentDist)
      else:
        features['enemyDistance'] = enemyDefenseAgentDist
        features['distanceToFood'] -= 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1
    else:
      foodList = self.getFood(successor).asList()
      if len(foodList) > 0: # This should always be True,  but better safe than sorry
        myPos = successor.getAgentState(self.index).getPosition()
        minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
        features['distanceToFood'] = minDistance



    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1, 'enemyDistance': 10, 'reverse': -2}

class DefInfRefAgent(InferenceReflexAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['invaderDistance'] = 0

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    for i in enemies: print i
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:


      if self.red:
        enemies = gameState.getBlueTeamIndices()
      else:
        enemies = gameState.getRedTeamIndices()
      #print enemies


      myPosition = successor.getAgentState(self.index).getPosition()

      if successor.getAgentPosition(enemies[0]):
        defensePosition = successor.getAgentPosition(enemies[0])
        enemyDefenseAgentDist = manhattanDistance(myPosition, defensePosition)
      else:
        enemyDefenseAgentDist = successor.getAgentDistances()[enemies[0]]
      print enemyDefenseAgentDist
      if enemyDefenseAgentDist <= 4:
        features['invaderDistance'] = enemyDefenseAgentDist

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}







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