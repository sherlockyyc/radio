

def log_write(file,log):
    """[将log写入文件中]

    Args:
        file ([file]): [要存储的文件]
        log ([dict]): [要保存的log]
    """
    string = ''
    for key, value in log.items():
        string += key + '\t:'+str(value) +'\n'
    string += '\n\n\n'
    file.write(string)

