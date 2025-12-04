
import numpy as np

class GRASPUTILS():
    def __init__(self):
        self.iou_tracker_left = TwoElementList()
        self.iou_tracker_right = TwoElementList()
        self.static_grasp_tracker = dict()
        self.left_overlap_vertices_tracker = TwoElementList()
        self.right_overlap_vertices_tracker = TwoElementList()

    def get_iou(self, previous_indices, current_indices):
        if len(previous_indices) == 0 or len(current_indices) == 0:
            return 0
        unique_elements = set(previous_indices).union(set(current_indices))
        binary_list1 = np.array([1 if elem in previous_indices else 0 for elem in unique_elements])
        binary_list2 = np.array([1 if elem in current_indices else 0 for elem in unique_elements])
        intersection = np.logical_and(binary_list1, binary_list2)
        union = np.logical_or(binary_list1, binary_list2)
        iou = np.sum(intersection) / np.sum(union)
        return iou
    
    def check_iou(self, iou_threshold=0.5):
        left_grasp = False
        right_grasp = False
        if self.iou_tracker_left.elements[0] > iou_threshold and self.iou_tracker_left.elements[1] > iou_threshold:
            left_grasp = True
        if self.iou_tracker_right.elements[0] > iou_threshold and self.iou_tracker_right.elements[1] > iou_threshold:
            right_grasp = True
        return left_grasp, right_grasp


class TwoElementList:
    def __init__(self, initial_elements=None):
        if initial_elements is None:
            initial_elements = []
        if len(initial_elements) > 2:
            raise ValueError("Initial list can't have more than two elements")
        self.elements = initial_elements[:2]
    
    def push(self, element):
        if len(self.elements) >= 2:
            self.elements.pop(0)
        self.elements.append(element)
    
    def pop(self):
        if not self.elements:
            raise IndexError("Pop from empty list")
        return self.elements.pop()
    
    def get_elements(self):
        return self.elements

    def __str__(self):
        lengths = [len(element) for element in self.elements]
        elements_info = [f"Element {i+1}: length={length}" for i, length in enumerate(lengths)]
        return "\n".join(elements_info)
