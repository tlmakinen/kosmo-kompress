# define inception block layer
def InceptBlock(filters, strides):
    """InceptNet convolutional striding block.
    filters: tuple: (f1,f2,f3)
    filters1: for conv1x1
    filters2: for conv1x1,conv3x3
    filters3L for conv1x1,conv5x5"""
    
    filters1, filters2, filters3 = filters
    
    conv1x1 = stax.serial(stax.Conv(filters1, (1,1), strides, padding="SAME"))
    
    conv3x3 = stax.serial(stax.Conv(filters1, (1,1), strides=None, padding="SAME"),
                        stax.Conv(filters2, (3,3), strides, padding="SAME"))
                        
    conv5x5 = stax.serial(stax.Conv(filters1, (1,1), strides=None, padding="SAME"),
                        stax.Conv(filters3, (5,5), strides, padding="SAME"))
                            

    return stax.serial(
          stax.FanOut(2),  # should num=3 or 2 here ?
          stax.parallel(conv1x1, conv3x3, conv5x5),
          stax.FanInConcat(), stax.LeakyRelu)

        

def InceptBlock2(filters, strides, dims=2, do_5x5=True, do_3x3=True):
    """InceptNet convolutional striding block.
    filters: tuple: (f1,f2,f3)
    filters1: for conv1x1
    filters2: for conv1x1,conv3x3
    filters3L for conv1x1,conv5x5"""
    
    filters1, filters2, filters3 = filters
    #strides = ()
    dim_nums = ("NCHWZ", "OIHWZ", "NCHWZ")
    
    conv1x1 = stax.serial(stax.GeneralConv(dim_nums, 
                                           filters1,
                                           filter_shape=(1,)*dims, 
                                           strides=strides, padding="SAME"))
    
    filters4 = filters2
    conv3x3 = stax.serial(stax.GeneralConv(dim_nums,
                                           filters2, 
                                           filter_shape=(1,)*dims, 
                                           strides=None, padding="SAME"),
                        stax.GeneralConv(dim_nums, 
                                         filters4,
                                         filter_shape=(3,)*dims, 
                                         strides=strides, padding="SAME"))
                        
    filters5 = filters3
    conv5x5 = stax.serial(stax.GeneralConv(dim_nums,
                                           filters3, 
                                           filter_shape=(1,)*dims, 
                                           strides=None, padding="SAME"),
                         stax.GeneralConv(dim_nums, 
                                          filters5, 
                                          filter_shape=(5,)*dims, 
                                          strides=strides, padding="SAME"))

    maxpool = stax.serial(stax.MaxPool((3,)*dims, padding="SAME"),
                         stax.GeneralConv(dim_nums, 
                                          filters4,
                                          filter_shape=(1,)*dims, 
                                          strides=strides, padding="SAME"))
                            
    if do_3x3:
        if do_5x5:
            return stax.serial(
                  stax.FanOut(4),  # should num=3 or 2 here ?
                  stax.parallel(conv1x1, conv3x3, conv5x5, maxpool),
                  stax.FanInConcat(), 
                  stax.LeakyRelu)
        else:
            return stax.serial(
                  stax.FanOut(3),  # should num=3 or 2 here ?
                  stax.parallel(conv1x1, conv3x3, maxpool),
                  stax.FanInConcat(), 
                  stax.LeakyRelu)
            
    else:
        return stax.serial(
              stax.FanOut(2),  # should num=3 or 2 here ?
              stax.parallel(conv1x1, maxpool),
              stax.FanInConcat(), 
              stax.LeakyRelu)
              
              
              
              
dim_nums = ("NCHWZ", "OIHWZ", "NCHWZ")

model = stax.serial(
        InceptBlock2((fs,fs,fs), strides=(4,4,4), dims=3, do_5x5=False),  # output: 8,8
        #InceptBlock2((fs,fs,fs), strides=(2,2,2), dims=3),  # output: 2,2
        InceptBlock2((fs,fs,fs), strides=(4,4,4), dims=3, do_5x5=False),  # output: 2,2
        #InceptBlock2((fs,fs,fs), strides=(2,2,2), dims=3, do_5x5=False, do_3x3=False), # output 2,2
        InceptBlock2((fs,fs,fs), strides=(2,2,2), dims=3, do_5x5=False, do_3x3=False), # output 1,1
        stax.GeneralConv(dim_nums, n_summaries, 
                         filter_shape=(1,1,1), 
                         strides=(1,1,1), padding="SAME"),
        stax.Flatten
)