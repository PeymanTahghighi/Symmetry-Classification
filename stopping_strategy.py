import numpy as np

class StoppingStrategy():
    def __init__(self, alpha) -> None:
        self._alpha = alpha;
        self._epsilon = 1e-5;
        pass
    
    def reset(self):
        pass

class GeneralizationLoss(StoppingStrategy):
    def __init__(self, alpha) -> None:
        super().__init__(alpha);
        self.__min = 99999;

    def __call__(self, val):
        #calculate actual metric
        gl = self._calc_metric(val);
        
        #decide to continue training or not
        if(gl > self._alpha):
            return False;
        return True;
    
    def _calc_metric(self, val):
        if(val < self.__min):
            self.__min = val;
        return 100 * ((val / self.__min) - 1);
    
    def reset(self):
        self.__min = 9999;

class CombinedTrainValid(StoppingStrategy):
    def __init__(self, alpha, strip) -> None:
        super().__init__(alpha);
        self.__strip = strip;
        self.__current_strip = 0;
        self.__generalization_loss = GeneralizationLoss(alpha);
        self.__sum_over_strip = 0;
        self.__min_over_strip = 999;
    
    def __call__(self, val, tr) -> bool:
        self.__update(tr);
        gl = self.__generalization_loss._calc_metric(val);

        #we check every strip times
        if self.__current_strip % self.__strip == 0:
            pk = 1000*((self.__sum_over_strip) / (self.__strip * self.__min_over_strip) - 1);
            pq = gl / (pk + self._epsilon);
            self.__current_strip = 0;
            self.__min_over_strip = 999;
            self.__sum_over_strip = 0;

            print(f"PQ: {pq}");
            if pq > self._alpha:
                return False;
            return True;
            
        return True;
            
    def __update(self, tr):
        #do calculation if we have not reached the decision making point
        self.__sum_over_strip += tr;
        #self.__generalization_loss._calc_metric(val);
        if self.__min_over_strip > tr:
            self.__min_over_strip = tr;
        self.__current_strip += 1;
    
    def reset(self):
        self.__current_strip = 0;
        self.__generalization_loss.reset();
        self.__sum_over_strip = 0;
        self.__min_over_strip = 999;


