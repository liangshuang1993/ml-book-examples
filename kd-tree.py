"""
In order to visualize, only suitable for 2-d data.
"""
import matplotlib.pyplot as plt
import copy

T = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]

class Tree(object):
	def __init__(self, node, d):
		self.left = None
		self.right = None
		self.node = node
		self.parent = None
		self.dim = d
		self.is_left = False

	def set_left_tree(self, child_tree):
		self.left = child_tree
		child_tree.parent = self
		child_tree.is_left = True

	def set_right_tree(self, child_tree):
		self.right = child_tree
		child_tree.parent = self


def build_kd_tree(data, d, square):
	data = sorted(data, key=lambda x:x[d])
	index = len(data)/2
	point = data[index]

	if d:
		# in this case, d=1 means y axis
		y1 = point[1]
		y2 = point[1]
		x1 = square[0][0]
		x2 = square[1][0]
	else:
		x1 = point[0]
		x2 = point[0]
		y1 = square[0][1]
		y2 = square[1][1]
	plt.plot([x1, x2], [y1, y2])
	tree = Tree(point, d)
	del data[index]

	# build left tree
	if index > 0:
		sub_square = copy.deepcopy(square)
		if d:
			sub_square[1][1] = point[1]
		else:
			sub_square[1][0] = point[0]
		tree.set_left_tree(build_kd_tree(data[: index], not d, sub_square))

	# build right tree
	if(len(data) > 1):
		sub_square = copy.deepcopy(square)
		if d:
			sub_square[0][1] = point[1]
		else:
			sub_square[0][0] = point[0]
		tree.set_right_tree(build_kd_tree(data[index: ], not d, sub_square))

	return tree

def get_distance(point1, point2):
	return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

num = 0

def search(target, tree, d):
	global num
	num += 1
	# if num > 6:
	# 	return
	if target[d] < tree.node[d]:
		if tree.left:
			return search(target, tree.left, not d)
	else:
		if tree.right:
			return search(target, tree.right, not d)
	min_distance = get_distance(target, tree.node)
	nearest_point = tree.node

	while tree.parent:
		node = tree.parent.node
		tmp_distance = get_distance(target, node)
		if tmp_distance < min_distance:
			min_distance = tmp_distance
			nearest_point = node
		# check if the circle cross split line
		d = tree.parent.dim
		if min_distance > abs(target[d] - node[d]):
			# check the other side
			if tree.is_left:
				if tree.parent.right:
					sub_tree = tree.parent.right
					sub_tree.parent = None
					result = search(target, sub_tree, sub_tree.dim)
					otherside_nearest_point = result[1]
					if(get_distance(nearest_point, target) > get_distance(otherside_nearest_point, target)):
						nearest_point = otherside_nearest_point
						min_distance = get_distance(otherside_nearest_point, target)
			else:
				if tree.parent.left:
					sub_tree = tree.parent.left
					sub_tree.parent = None
					result = search(target, sub_tree, sub_tree.dim)
					otherside_nearest_point = result[1]
					if(get_distance(nearest_point, target) > get_distance(otherside_nearest_point, target)):
						nearest_point = otherside_nearest_point
						min_distance = get_distance(otherside_nearest_point, target)

		tree = tree.parent
	return min_distance, nearest_point
			



fig = plt.figure(figsize=(6, 6))
plt.axes(xlim=(0, 10), ylim=(0, 10))
T_x = []
T_y = []
for t in T:
	T_x.append(t[0])
	T_y.append(t[1])
plt.plot(T_x, T_y, 'bo')
kd_tree = build_kd_tree(T, 0, [[0, 0], [10, 10]])


# input a point
point = [8, 3]
result = search(point, kd_tree, 0)
print result

plt.plot(point[0],point[1], 'rx')
plt.plot(result[1][0],result[1][1], 'r^')

plt.show()