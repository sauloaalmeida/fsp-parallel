from numpy.random import MT19937
from numpy.random import Philox
from numpy.random import RandomState

rs = RandomState(42)

#mt19937 = MT19937()
#mt19937.state = rs.get_state()
#rs2 = RandomState(mt19937)

#philox = Philox()
#philox.state = rs.get_state()
#rs3 = RandomState(philox)


# Same output
print(rs.standard_normal())
#print(rs2.standard_normal())
#print(rs3.standard_normal())

print(rs.random())
print(rs.random())
#print(rs2.random())
#print(rs3.random())
#print(rs.standard_exponential())
#print(rs2.standard_exponential())