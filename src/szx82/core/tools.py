def slice(start_stop, data):
    start_stop = list(start_stop)

    if start_stop[0] is not None \
        and start_stop[0] !=0 and start_stop[0] < 1: 
        start_stop[0] = int(len(data) * start_stop[0])
    if start_stop[1] is not None \
        and start_stop[1] !=0 and start_stop[1] < 1:
        start_stop[1] = int(len(data) * start_stop[1])

    return tuple(start_stop)
