def insert_before_index(lst, index, new_element):
    """
    Insert a new element before the specified index in the list

    :param lst: Original list
    :param index: Index of the specified element
    :param new_element: New element to insert
    """
    if 0 <= index < len(lst):
        lst.insert(index, new_element)
    else:
        print(f"Index {index} is out of range")
    return lst


def insert_after_index(lst, index, new_element):
    """
    Insert a new element after a specified index in the list

    :param lst: Original list
    :param index: Specified index
    :param new_element: New element to insert
    """
    # Check if the index is within the bounds of the list
    if index < 0 or index >= len(lst):
        print(f"Index {index} is out of bounds")
        return lst

    # Insert the new element after the specified index
    lst.insert(index + 1, new_element)
    return lst


def count_occurrences_up_to_index(lst, target, index):
    """
    Count how many times the target element has appeared in the list up to the given index

    :param lst: Original list
    :param target: Target element
    :param index: Specified index
    :return: The count of occurrences of the target element up to the specified index
    """
    if index >= len(lst):
        raise IndexError("Index is out of the bounds of the list")

    count = 0
    for i in range(index + 1):
        if lst[i] == target:
            count += 1
    return count


def find_nth_occurrence(lst, element, n):
    """
    Find the index of the nth occurrence of an element in a list.

    :param lst: The list to search.
    :param element: The element to find.
    :param n: The occurrence number (1-based).
    :return: The index of the nth occurrence of the element, or -1 if not found.
    """
    count = 0
    for index, current_element in enumerate(lst):
        if current_element == element:
            count += 1
            if count == n:
                return index
    return -1


def get_surrounding_elements(lst, index, n=4):
    """
    Get the surrounding elements and their indices from the list.

    :param lst: The list to search.
    :param index: The target index.
    :param n: The number of elements to retrieve before and after the target index.
    :return: A list of tuples containing (index, element).
    """
    start_index = max(0, index - n)
    end_index = min(len(lst), index + n + 1)

    surrounding_elements = []

    for i in range(start_index, end_index):
        if i != index:
            surrounding_elements.append((lst[i], i))

    return surrounding_elements


def get_surrounding_elements_contain_itself(lst, index, n=4):
    """
    Get the surrounding elements and their indices from the list, including the target index.

    :param lst: The list to search.
    :param index: The target index.
    :param n: The number of elements to retrieve before and after the target index.
    :return: A list of tuples containing (element, index).
    """
    start_index = max(0, index - n)
    end_index = min(len(lst), index + n + 1)

    surrounding_elements = []

    for i in range(start_index, end_index):
        surrounding_elements.append((lst[i], i))

    return surrounding_elements


def replace_elements(lst, target, replacement):
    """
    Replace all occurrences of the target element in the list with the replacement element.

    :param lst: The list of elements.
    :param target: The element to be replaced.
    :param replacement: The element to replace with.
    :return: A new list with the target elements replaced.
    """
    return [replacement if element == target else element for element in lst]


def replace_element_by_index(lst, index, new_value):
    """
    Replace the element at the specified index with a new value.

    :param lst: The list to modify.
    :param index: The index of the element to replace.
    :param new_value: The new value to set at the specified index.
    :return: The modified list.
    """
    if 0 <= index < len(lst):
        lst[index] = new_value
    else:
        raise IndexError("Index out of range")
    return lst
