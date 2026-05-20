class NoSpaceError(Exception):
    def __init__(self, msg=None):
        super().__init__(msg or "Queue not enough space")

class circular_queue:
    def __init__(self, n):
        self.n = n
        self._queue = [None]  * n
        self._head = 0
        self._tail = 0
        self._count = 0

    def _append_el(self, e):
        self._queue[self._tail] = e
        self._tail = (self._tail + 1) % self.n
        self._count += 1

    def add(self, element):
        if self._count == self.n:
            raise NoSpaceError()
        
        self._append_el(element)        
        return True
    
    def element(self):
        if self._count == 0:
            raise ValueError('queue is empty')
        
        return self._queue[self._head]
    
    def offer(self, element):
        if self._count == self.n:
            return False
        
        self._append_el(element)
        return True
    
    def peek(self):
        if self._count == 0:
            return None
        return self._queue[self._head]
    
    def _remove_el(self):
        res = self._queue[self._head]
        self._head = (self._head + 1) % self.n
        self._count -= 1
        return res 
    
    def poll(self):
        if self._count == 0:
            return None
        
        return self._remove_el()
    
    def remove(self):
        if self._count == 0:
            raise ValueError('queue is empty')

        return self._remove_el()

    def size(self):
        return self._count
    
    def __str__(self):
        return '[' + ', '.join(str(self._queue[i % self.n]) 
                               for i in range(self._head, self._head + self._count)) + ']'
    

queue = circular_queue(2)

queue.add("apple")
queue.add("banana")
queue.add("cherry")

# print the queue
print("Queue: ", queue)

# remove the element at the front of the queue
front = queue.remove()
print("Removed element: ", front)

# print the updated queue
print("Queue after removal: ", queue)

# add another element to the queue
queue.add("date")

# peek at the element at the front of the queue
peeked = queue.peek()
print("Peeked element: ", peeked)

# print the updated queue
print("Queue after peek: ", queue)
# Резултат

# Queue: [apple, banana, cherry]
# Removed element: apple
# Queue after removal: [banana, cherry]
# Peeked element: banana
# Queue after peek: [banana, cherry, date]
#=========================================
# q = circular_queue(10)

# a = [0, 1, 2, 3, 4]

# for i in a:
#     q.add(i)
#     # q.offer(i)

# # display contents of the queue.
# print("Elements of queue ", q)

# # remove the head of queue.
# removedele = q.remove()
# print("removed element-", removedele)

# print(q)

# # head of queue
# head = q.peek()
# print("head of queue-", head)

# size = q.size()
# print("Size of queue-", size)
# Резултат

# Elements of queue [0, 1, 2, 3, 4]
# removed element-0
# [1, 2, 3, 4]
# head of queue-1
# Size of queue-4