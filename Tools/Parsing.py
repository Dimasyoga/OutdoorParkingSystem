import json
import dpath.util
import time
import logging

logging.getLogger().setLevel(logging.INFO)
class Parsing(object):
    def __init__(self):
        self.json_raw = None
        self.conf_json_raw = None
        self.cam_status = {}
        self.cam_path = []
        self.url_list = []
        self.masking_list = []
        self.slot_path = []
        self.slot_reserved = []
    
    def update(self):
        self.cam_status = {}
        self.cam_path = []
        self.url_list = []
        self.masking_list = []
        self.slot_path = []
        self.slot_reserved = []
        slot_count = 0
        cam_count = 0

        for (loc, cam) in self.json_raw.items():
            for (k, v) in cam.items():
                cam_count += 1
                self.url_list.append(v['url'])
                self.cam_path.append(loc + '/' + k)
                path = loc + '/' + k
                dpath.util.new(self.cam_status, path+'/status', False)
                dpath.util.new(self.cam_status, path+'/free', 0)
                cam_mask = []
                slot = []
                reserved = []
                for (key, value) in v['slot'].items():
                    slot_count += 1
                    path = loc + '/' + k + '/slot/' + key
                    slot.append(path)
                    message = {}
                    message['free'] = False
                    reserved.append(value['reserved'])
                    dpath.util.new(self.cam_status, path, message)
                    slot_mask = value['masking']
                    x = []
                    for point in slot_mask:
                        x.append(tuple(point))
                    cam_mask.append(x)
                self.masking_list.append(cam_mask)
                self.slot_path.append(slot)
                self.slot_reserved.append(reserved)
        
        logging.info(f"Registered cam: {cam_count}, Total slot: {slot_count}")
        # print(self.slot_path)

    def input_config(self, config):
        """
        docstring
        """
        self.json_raw = config

        for (loc, cam) in self.json_raw.items():
            for (k, v) in cam.items():
                self.url_list.append(v['url'])
                self.cam_path.append(loc + '/' + k)
                path = loc + '/' + k
                dpath.util.new(self.cam_status, path+'/status', False)
                dpath.util.new(self.cam_status, path+'/free', 0)
                cam_mask = []
                slot = []
                reserved = []
                for (key, value) in v['slot'].items():
                    path = loc + '/' + k + '/slot/' + key
                    slot.append(path)
                    message = {}
                    message['free'] = False
                    reserved.append(value['reserved'])
                    dpath.util.new(self.cam_status, path, message)
                    slot_mask = value['masking']
                    x = []
                    for point in slot_mask:
                        x.append(tuple(point))
                    cam_mask.append(x)
                self.masking_list.append(cam_mask)
                self.slot_path.append(slot)
                self.slot_reserved.append(reserved)

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
        logging.info(f"System config is {self.conf_json_raw}")

    def get_start_time(self):
        return self.conf_json_raw['start_time']

    def get_end_time(self):
        return self.conf_json_raw['end_time']
    
    def get_update_rate(self):
        return self.conf_json_raw['update_rate']
    
    def get_threshold(self):
        return self.conf_json_raw['threshold']
    
    def get_cam_timeout(self):
        return self.conf_json_raw['cam_timeout']

    def get_url(self):
        return self.url_list
    
    def get_masking(self):
        return self.masking_list
    
    def get_slot_path(self):
        return self.slot_path
    
    def get_slot_reserved(self):
        return self.slot_reserved
    
    def input_status(self, inp):
        # res = {}
        for index, result, total_free in inp:
            dpath.util.merge(self.cam_status, result, flags=(1 << 1))
            # dpath.util.set(self.cam_status, '/', result)
            # res.update(result)
            # print(result)
            # dpath.util.set(self.cam_status, self.cam_path[index]+'/status', status)
            # dpath.util.set(self.cam_status, self.cam_path[index]+'/free', total_free)
        # print(res)

    def get_free(self):
        output = json.loads(json.dumps(self.cam_status))
        dpath.util.new(output, 'last_update', time.ctime(time.time()))

        return output
    
    def config_ready(self):
        if (self.cam_status != {}) and (self.conf_json_raw != None):
            return True
        else: 
            return False


        
# if __name__ == "__main__":
#     pars = Parsing()

#     with open("config.json", 'r') as f:
#         data = json.load(f)
    
#     pars.input_config(data)
#     print(pars.get_url())
#     print(pars.get_masking())