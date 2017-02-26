from threading import Thread

def t1_Task(list1, list2):
    for i in list1:
        print "t1: {0}".format(i)
        list2.append(i)

def t2_Task(list1, list2):
    for i in list1:
        print "t2: {0}".format(i)
        list2.append(i)

def main():
    list1 = [1, 2, 3, 4, 5]
    list2 = []
    thread1 = Thread(target = t1_Task, args=(list1, list2, ))
    thread2 = Thread(target = t2_Task, args=(list1, list2, ))
    thread1.start()
    thread2.start()
    thread1.join()
    print list2

main()
