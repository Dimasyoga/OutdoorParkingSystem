import json
import dpath.util
import time
import logging

logging.getLogger().setLevel(logging.INFO)
class Parsing(object):
    """
    A class used to handle data parsing with database

    Attributes
    ----------
    json_raw : dict
        a dict of camera properties (url, masking parameter, etc)
    conf_json_raw : dict
        a dict of system config (update time, threshold, working time, etc)
    cam_status : dict
        a dict of cam status
    cam_path : list
        a list of cam dict path in cam_status
    url_list : list
        a list of camera url
    masking_list: list
        a list of masking parameter list for each camera
    slot_path : list
        a list of slot path in cam_status for each camera
    slot_reserved : list
        a list of reserved flag for each slot of each camera

    """
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
        """Parsing json raw to class attribute"""
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

    def input_config(self, config):
        """Input json_raw manually"""
        self.json_raw = config
        self.update()

    def stream_handler(self, message):
        """
        Handle stream from database for camera properties
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
        Handle stream from database for system config
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
        """Get system start update time

        Returns
        -------
        time : int
            Time for system to start updating in hours (24 hours format)
        """
        return self.conf_json_raw['start_time']

    def get_end_time(self):
        """Get system end update time

        Returns
        -------
        time : int
            Time for system to start updating in hours (24 hours format)
        """
        return self.conf_json_raw['end_time']
    
    def get_update_rate(self):
        """Get system update rate

        Returns
        -------
        rate : int
            Minimal time gap for system to do update process in second
        """
        return self.conf_json_raw['update_rate']
    
    def get_threshold(self):
        """Get system detection threshold

        Returns
        -------
        threshold : float
            detection threshold of inference result to determine slot status
        """
        return self.conf_json_raw['threshold']
    
    def get_cam_timeout(self):
        """Get system cam timeout

        Returns
        -------
        timeout : float
            http request timeout of request image from camera
        """
        return self.conf_json_raw['cam_timeout']

    def get_url(self):
        """Get list of camera url

        Returns
        -------
        url_list : list
            list of camera url
        """
        return self.url_list
    
    def get_masking(self):
        """Get masking parameter of all camera

        Returns
        -------
        masking_list : list
            list of masking parameter
        """
        return self.masking_list
    
    def get_slot_path(self):
        """Get slot_path """
        return self.slot_path
    
    def get_slot_reserved(self):
        """Get slot_reserved """
        return self.slot_reserved
    
    def input_status(self, inp):
        """Set cam_status attribute from update process result

        Parameters
        ----------
        inp : list
            List of process result for each camera
        """
        for index, result, total_free in inp:
            dpath.util.merge(self.cam_status, result, flags=(1 << 1))

    def get_free(self):
        """Get output dictionary for uploading to database

        Returns
        -------
        output : dict
            Class attribute cam_status with last_update info
        """
        output = json.loads(json.dumps(self.cam_status))
        dpath.util.new(output, 'last_update', time.ctime(time.time()))

        return output
    
    def config_ready(self):
        """Get system config availability (cam_status and conf_json_raw)

        Returns
        -------
        status : bool
        """
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