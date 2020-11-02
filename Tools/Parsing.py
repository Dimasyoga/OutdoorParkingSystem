import json
import dpath.util
import time

class Parsing(object):
    def __init__(self):
        self.json_raw = None
        self.conf_json_raw = None
        self.parking_lot = {}
        self.cam_status = []
        self.free_cam = []
        self.url_list = []
        self.masking_list = []
    
    def update(self):
        self.parking_lot = {}
        self.cam_status = []
        self.free_cam = []
        self.url_list = []
        self.masking_list = []

        for (k, v) in self.json_raw.items():
            self.parking_lot[k] = len(self.json_raw[k])
            for (key, value) in v.items():
                self.url_list.append(value['url'])
                self.masking_list.append(value['masking'])
                self.cam_status.append(False)
                self.free_cam.append(0)
        
    def input_config(self, config):
        """
        docstring
        """
        self.json_raw = config

        for (k, v) in config.items():
            self.parking_lot[k] = len(config[k])
            for (key, value) in v.items():
                self.url_list.append(value['url'])
                self.masking_list.append(value['masking'])
                self.cam_status.append(False)
                self.free_cam.append(0)

    def stream_handler(self, message):
        """
        docstring
        """
        path = message["path"][1:]
        if (message["data"] != None):
            if (path == ""):
                self.json_raw = message["data"]
            else:
                dpath.util.new(self.json_raw, path, message["data"])
        else:
            dpath.util.delete(self.json_raw, path)
        
        self.update()
    
    def config_handler(self, message):
        """
        docstring
        """
        path = message["path"][1:]
        if (message["data"] != None):
            if (path == ""):
                self.conf_json_raw = message["data"]
            else:
                dpath.util.new(self.conf_json_raw, path, message["data"])
        else:
            dpath.util.delete(self.conf_json_raw, path)
        print(self.conf_json_raw)

    def get_start_time(self):
        return self.conf_json_raw["start_time"]

    def get_end_time(self):
        return self.conf_json_raw["end_time"]
    
    def get_update_rate(self):
        return self.conf_json_raw["update_rate"]
    
    def get_free_threshold(self):
        return self.conf_json_raw["free_threshold"]
    
    def get_cam_timeout(self):
        return self.conf_json_raw["cam_timeout"]

    def get_url(self):
        return self.url_list
    
    def get_masking(self):
        
        result = []
        for cam in self.masking_list:
            mask = []
            for slot in cam:
                x = []
                for point in slot:
                    x.append(tuple(point))
                mask.append(x)
            result.append(mask)
        
        return result
    
    def input_status(self, inp):
        for idx, stat, free_cam in inp:
            if stat:
                self.cam_status[idx] = True
                self.free_cam[idx] = free_cam
            else:
                self.cam_status[idx] = False
        
    def get_free_lot_total(self):
        output = {}
        count = 0
        for (k, v) in self.parking_lot.items():
            sum = 0
            for idx in range(count, count+v):
                sum = sum + self.free_cam[idx]
            output[k] = sum
            count = count + v
        
        output["last_update"] = time.ctime(time.time())
        
        return output

    def get_free_lot_all(self):
        output = {}
        count = 0
        for (k, v) in self.json_raw.items():
            out = {}
            for (key, value) in v.items():
                out[key] = {"cam_status": self.cam_status[count],"free_space": self.free_cam[count]}
                count = count + 1
            output[k] = out
        
        output["last_update"] = time.ctime(time.time())
        
        return output
    
    def all_cam_true(self):
        ret = True
        for stat in self.cam_status:
            if not stat:
                ret = False
        
        return ret
    
    def get_false_cam_index(self):
        """
        docstring
        """
        idx = []
        for i, stat in enumerate(self.cam_status):
            if not stat:
                idx.append(i)
        
        return idx
    
    def config_ready(self):
        if (self.json_raw != None) and (self.conf_json_raw != None):
            return True
        else: 
            return False


        
# if __name__ == "__main__":
    # pars = Parsing()

    # with open("config.json", 'r') as f:
    #     data = json.load(f)
    
    # pars.input_config(data)
    # print(pars.get_url())
    # print(pars.get_masking())
    # print(pars.get_free_lot_total())
    # print(pars.get_free_lot_all())