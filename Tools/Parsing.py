import json
import dpath.util
import time

class Parsing(object):
    def __init__(self):
        self.json_raw = None
        self.conf_json_raw = None
        self.cam_status = {}
        self.cam_path = []
        self.url_list = []
        self.masking_list = []
        self.slot_path = []
    
    def update(self):
        self.cam_status = {}
        self.cam_path = []
        self.url_list = []
        self.masking_list = []
        self.slot_path = []

        for (loc, cam) in self.json_raw.items():
            for (k, v) in cam.items():
                self.url_list.append(v['url'])
                self.cam_path.append(loc + '/' + k)
                path = loc + '/' + k + '/status'
                dpath.util.new(self.cam_status, path, False)
                cam_mask = []
                slot = []
                for (key, value) in v['slot'].items():
                    path = loc + '/' + k + '/slot/' + key
                    slot.append(path)
                    message = {}
                    message['free'] = False
                    message['reserved'] = value['reserved']
                    dpath.util.new(self.cam_status, path, message)
                    slot_mask = value['masking']
                    x = []
                    for point in slot_mask:
                        x.append(tuple(point))
                    cam_mask.append(x)
                self.masking_list.append(cam_mask)
                self.slot_path.append(slot)
        
    def input_config(self, config):
        """
        docstring
        """
        self.json_raw = config

        for (loc, cam) in self.json_raw.items():
            for (k, v) in cam.items():
                self.url_list.append(v['url'])
                self.cam_path.append(loc + '/' + k)
                path = loc + '/' + k + '/status'
                dpath.util.new(self.cam_status, path, False)
                cam_mask = []
                slot = []
                for (key, value) in v['slot'].items():
                    path = loc + '/' + k + '/slot/' + key
                    slot.append(path)
                    message = {}
                    message['free'] = False
                    message['reserved'] = value['reserved']
                    dpath.util.new(self.cam_status, path, message)
                    slot_mask = value['masking']
                    x = []
                    for point in slot_mask:
                        x.append(tuple(point))
                    cam_mask.append(x)
                self.masking_list.append(cam_mask)
                self.slot_path.append(slot)

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
        return self.conf_json_raw['start_time']

    def get_end_time(self):
        return self.conf_json_raw['end_time']
    
    def get_update_rate(self):
        return self.conf_json_raw['update_rate']
    
    def get_free_threshold(self):
        return self.conf_json_raw['free_threshold']
    
    def get_cam_timeout(self):
        return self.conf_json_raw['cam_timeout']

    def get_url(self):
        return self.url_list
    
    def get_masking(self):
        return self.masking_list
    
    def input_status(self, inp):
        for idx, stat, free_cam in inp:
            dpath.util.set(self.cam_status, self.cam_path[idx]+'/status', stat)
            # slot_path = []
            # for i, (k, v) in enumerate(dpath.util.get(self.cam_status, self.cam_path[idx])['slot'].items()):
            #     # slot_path.append(self.cam_path[idx]+'/slot/'+k)
            #     dpath.util.set(self.cam_status, self.cam_path[idx]+'/slot/'+k+'/free', free_cam[i])
            for i,slot_stat in enumerate(free_cam):
                dpath.util.set(self.cam_status, self.slot_path[idx][i]+'/free', slot_stat)
        
    def get_free(self):
        output = {}
        total_free = 0
        for (loc, cam) in self.cam_status.items():
            lot_free = 0
            for (k, v) in cam.items():
                cam_free = 0
                for (key, value) in v['slot'].items():
                    dpath.util.new(output, loc+'/'+k+'/slot/'+key, dpath.util.get(self.cam_status, loc+'/'+k+'/slot/'+key))
                    if value['free'] and not value['reserved']:
                        cam_free += 1
                lot_free += cam_free
                dpath.util.new(output, loc+'/'+k+'/free', cam_free)
            total_free += lot_free
            dpath.util.new(output, loc+'/free', lot_free)
        
        dpath.util.new(output, 'free', total_free)
        dpath.util.new(output, 'last_update', time.ctime(time.time()))

        return output
    
    def config_ready(self):
        if (self.json_raw != None) and (self.conf_json_raw != None):
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