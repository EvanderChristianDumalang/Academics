#!/bin/python

import math
import os
import random
import re
import sys



n = 20
if (n % 2 == 1):
    print("Weird")
elif n in range (2,5) and (n % 2 == 0):
    print("Not Weird")
elif n in range (6,20) and (n % 2 == 0):
    print("Weird")
elif (n % 2 == 0 and n > 20):
    print("Not Weird")