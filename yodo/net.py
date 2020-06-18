import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.layers import Input,Conv2D,MaxPool2D,Activation,Reshape,BatchNormalization,concatenate
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint,LearningRateScheduler,Callback

from .utils import JSON
from . import np

def block4X(_in,filters,ksize,name,pool=False):
    conv0 = Conv2D(filters,ksize,activation="relu",padding="same",name=f"{name}_conv0")(_in)
    conv1 = Conv2D(filters,ksize,activation="relu",padding="same",name=f"{name}_conv1")(conv0)
    batch1 = BatchNormalization(name=f"norm0_{name}")(conv1)
    conv2 = Conv2D(filters,ksize,activation="relu",padding="same",name=f"{name}_conv2")(batch1)
    conv3 = Conv2D(filters,ksize,activation="relu",padding="same",name=f"{name}_conv3")(conv2)
    batch2 = BatchNormalization(name=f"norm1_{name}")(conv3)
    out  = concatenate([conv0,conv1,conv2,batch2],name=f"{name}_conc")
    
    if pool:
        out = MaxPool2D(name=f"{name}_pool")(out)
    return out

def block2X(_in,filters,ksize,name,pool=False):
    conv0 = Conv2D(filters,ksize,activation="relu",padding="same",name=f"{name}_conv0")(_in)
    conv1 = Conv2D(filters,ksize,activation="relu",padding="same",name=f"{name}_conv1")(conv0)
    conc = concatenate([conv0,conv1],name=f"{name}_conc") 
    out = BatchNormalization(name=f"{name}_norm")(conc)
    
    if pool:
        out = MaxPool2D(name=f"{name}_pool")(out)
    return out


def proposals(_in,k,name):
    prob = Conv2D(k*2,1,padding="same",name=f"prob_conv_{name}")(_in)
    prob = BatchNormalization(name=f"prob_norm_{name}")(prob)
    prob = Activation("sigmoid",name=f"prob_out_{name}")(prob)
    prob = Reshape((-1,2),name=f"{name}_prob")(prob)

    box = Conv2D(k*4,1,padding="same",name=f"box_conv_{name}")(_in)
    box = BatchNormalization(name=f"box_batch_{name}")(box)
    box = Activation("sigmoid",name=f"box_out_{name}")(box)
    box = Reshape((-1,4),name=f"{name}_box")(box)
    
    return prob,box

def get_split(length:int,test_size):
    test = test_size if isinstance(length,int) else int(test_size*length)
    train_index = np.arange(0,length)
    test_index = set(np.random.choice(train_index,test,replace=False))
    train_index = set(train_index)    
    train_index = list(train_index.difference(test_index))
    test_index = list(test_index)

    return train_index,test_index

def flow(epochs:int,x:list,y:tuple,index:list):
    images = x
    y_prob,y_boxes = y
    for _ in range(epochs):
        for i in index:
            yield images[i:i+1],(y_prob[i:i+1],y_boxes[i:i+1])

class BoxLoss(tf.Module):
    """
    BoxLoss
    """
    def __init__(self,):
        self.__name__="BoxLoss"
        self.zero = tf.constant(0,tf.float32)
        self.one = tf.constant(1,tf.float32)
        self.two = tf.constant(2,tf.float32)
        self.l2_loss = tf.keras.losses.MeanSquaredError()

        self.slice_y = {"begin":[0,0,0],"size":[-1,-1,1]}
        self.slice_x = {"begin":[0,0,1],"size":[-1,-1,1]}
        self.slice_h = {"begin":[0,0,2],"size":[-1,-1,1]}
        self.slice_w = {"begin":[0,0,3],"size":[-1,-1,1]}
        
        self.repr = f"""
BoxLoss(
    
)
        """
        
    def __repr__(self,):
        return self.repr

    
    @tf.function
    def __call__(self,y_true,y_pred,*args,**kwargs):
        
        y = tf.slice(y_true,**self.slice_y)
        x = tf.slice(y_true,**self.slice_x)

        mask = tf.logical_or(tf.greater(y,self.zero),tf.greater(x,self.zero))

        y = y[mask] 
        x = x[mask] 
        h = tf.slice(y_true,**self.slice_h)[mask] 
        w = tf.slice(y_true,**self.slice_w)[mask] 

        y_ = tf.slice(y_pred,**self.slice_y)[mask] 
        x_ = tf.slice(y_pred,**self.slice_x)[mask] 
        h_ = tf.slice(y_pred,**self.slice_h)[mask] 
        w_ = tf.slice(y_pred,**self.slice_w)[mask] 
        
        w2  = tf.divide(w,self.two)
        h2  = tf.divide(h,self.two)
        w2_  = tf.divide(w_,self.two)
        h2_  = tf.divide(h_,self.two)
        
        xmin = tf.subtract(x,w2)
        xmax = tf.add(x,w2)
        ymin = tf.subtract(y,h2)
        ymax = tf.add(y,h2)
        
        xmin_ = tf.subtract(x_,w2_)
        xmax_ = tf.add(x_,w2_)
        ymin_ = tf.subtract(y_,h2_)
        ymax_ = tf.add(y_,h2_)
        
        Cymin = tf.maximum(ymin,ymin_)
        Cymax = tf.minimum(ymax,ymax_)
        Cxmin = tf.maximum(xmin,xmin_)
        Cxmax = tf.minimum(xmax,xmax_)

        Ch = tf.subtract(Cymax , Cymin)
        Cw = tf.subtract(Cxmax , Cxmin)
        
        Aa = tf.multiply( h , w )
        Ba = tf.multiply( h_ , w_ )
        Ca = tf.multiply( Cw , Ch )
        
        IoU = tf.reduce_mean(tf.subtract(self.one,tf.divide(Ca,tf.subtract(tf.add(Aa,Ba),Ca))))
        Xl2 = self.l2_loss(x,x_)
        Yl2 = self.l2_loss(y,y_)
        
        loss = Xl2 + Yl2 + IoU
        if tf.math.is_nan(loss):
            return self.zero
        return loss

class YODO(object):
    """
    Builder Class
    """
    def __init__(self,config):
        assert isinstance(config,JSON),"Please use JSON or Config object for config."
        self.config = config
        self.builder_config = {
                "block4X":block4X,
                "block2X":block2X
                }
        self.loss = {
            "prob":keras.losses.BinaryCrossentropy(),
            "box":BoxLoss()
        }
        
    def build(self,optimizer=None):
        optimizer = optimizer if optimizer else keras.optimizers.SGD() 
        _in = Input((self.config.img_size,self.config.img_size,3))
        last = _in
        for b in self.config.backbone:
            b = self.builder_config[b.block](_in=last,**b.params())
            last = b
        self.backbone = keras.Model(_in,last,name="Backbone")        
        prob_out,box_out = [],[]
        for i,p in enumerate(self.config.proposals):
            features = block4X(last,**p.params())
            p,b = proposals(features,p.k,f"proposals_{i}")
            prob_out.append(p)
            box_out.append(b)
            last = features

        prob = concatenate(prob_out,axis=1,name="prob")
        box = concatenate(box_out,axis=1,name="box")
        self.model = keras.Model(_in,[prob,box],name="YODO")
        self.model.compile(optimizer=optimizer,loss=self.loss)
        
            
    def summary(self):
        self.model.summary()
            
    def train(self,x,y,epochs,batch_size,callbacks=[]):
        assert len(y) == 2,"Please pass probablity and encoded boxes as a tuple"
        self.model.fit(x,y,batch_size=batch_size,epochs=epochs,callbacks=callbacks)
        
    def train_generator(self,flow,steps_per_epoch,epochs,callbacks=[]):
        self.model.fit_generator(flow,steps_per_epoch=steps_per_epoch,epochs=epochs,callbacks=callbacks)

