# Vision2.0

Vision 2.0 is an Image-Processing based Robotics Competition being organized by the Robotics Club, IIT (BHU), Varanasi to facilitate learning about different components of image processing and its application in building robots capable of autonomous movement. In this online semester, as all of us were not present physically in the campus, conducting a physical robotics competition was not feasible. Hence, this year the event will be held online using PyBullet - a python module for physics simulations of robots.
<br>The official arena for the event can be found on - https://github.com/Robotics-Club-IIT-BHU/Vision-2.0-2020-Arena<br>
**The problem statement of the event can be found here: [Vision2.0Problem Statement.pdf](https://github.com/AnandSidd/Vision2.0/blob/master/Vision2.0%20Problem%20Statement-1.pdf)**

# Vision-2.0 Problem Statement

![](arena.png)

# Arena Description:
 There are 2 paths (inner and outer square) and there are 4
connecting paths of different colours joining them.

 Bot can change from outer path to inner path or vice versa. Bot
is allowed to move in a clockwise direction only. The portion of
the arena in black colour is restricted for the movement of the
bot.

 There will be 3 shapes (square, circle and triangle) of 2
different colours, distinguishing each block in 6 different ways.
All shape colours might change but they will be distinguishable
easily. The position of shapes in the final area will not be the
same as the indicative pictures

 On the outermost path there will be 4 arrows at the end of
connecting paths pointing in clockwise direction. These arrows
mark the Starting Zone where the bot will be placed initially on
any one of the arrows.

 The Centre of the arena is the home zone.

 The bot has to traverse the arena, complete a full round and
finish at the home zone.

 A video feed from the overhead camera will be provided to the
team. The team’s computer should autonomously instruct their
bot throughout the arena using this feed.
# Game Procedure:
1. The bot is placed at one of the Starting Zones.
2. Team will be given an abbreviation which associate to
specific colour and shape.
RT for Red Triangle.
RS for Red Square.
RC for Red Circle.
YT for Yellow Triangle.
YS for Yellow Square.
YC for Yellow Circle.
3. The bot must then find the closest block which it can reach
following a clockwise path. If two positions with the
required colour and shape are at the same distance from
the bot then the bot may choose either.
4. Signal must be sent to when bot stops moving.
5. As soon as the bot stops moving, bot has to ask for input
using the function provided.
6. This continues till bot has completed a full round around
the center, Then it should move to home via the
connecting paths that it started on.
7. On reaching home the bot should signal that it has finished
the task.
# Our Approach
1. We used opencv(An opensource image processing library) to segment out different colour with their respective shape.
2. Made a 2d array by giving unique numbers to each shape with its colour.
3. We used BFS as a path finding algorithm.
4. Then an optimal graph/path was plotted to get the optimal trajectory, connecting the centroids of all the shapes along the way to the destination shape.

