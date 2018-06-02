# coding:utf-8

import os
import sys
import signal
import time
import shutil
import subprocess
import fasteners
import numpy as np
import gym
from gym import spaces

from abc import ABCMeta, abstractmethod

######################################################
class VrepEnv:
    def __init__(self, scene="rollbalance", is_render=True, is_boot=True, port=19997):
        # import V-REP
        if "linux" in sys.platform:
            self.VREP_DIR = os.path.expanduser("~") + "/V-REP_PRO_EDU/"
            exeDir = ""
            vrepExe = "vrep.sh"
            libExe = "Linux/64Bit/remoteApi.so"
        elif "darwin" in sys.platform:
            self.VREP_DIR = "/Applications/V-REP_PRO_EDU/"
            exeDir = "vrep.app/Contents/MacOS/"
            vrepExe = "vrep"
            libExe = "Mac/remoteApi.dylib"
        else:
            print(sys.platform)
            sys.stderr.write("I don't know how to use vrep in Windows...\n")
            sys.exit(-1)
        # copy remote api library to the accesible directory
        remotePath = self.VREP_DIR + "programming/remoteApiBindings/"
        sys.path.append(remotePath + "python/python/")
        try:
            remoteFile = remotePath + "python/python"+libExe[libExe.rfind("/"):]
            if not os.path.exists(remoteFile):
                print(remoteFile + "is not found, so copied from lib directory.")
                shutil.copyfile(remotePath + "lib/lib/" + libExe, remoteFile)
        except:
            pass
        # load remote api library
        try:
            global vrep
            import vrep
        except:
            print ('--------------------------------------------------------------')
            print ('"vrep.py" could not be imported. This means very probably that')
            print ('either "vrep.py" or the remoteApi library could not be found.')
            print ('Make sure both are in the same folder as this file,')
            print ('or appropriately adjust the file "vrep.py"')
            print ('--------------------------------------------------------------')
            sys.exit(-1)
        # confirm scene file and change mode depending on the scene
        # scenePath = os.path.abspath(os.path.dirname(__file__))+"/scenes/"+scene+".ttt"
        scenePath = os.path.abspath(os.path.dirname(__file__)) + "/scenes/"
        modeName = "normal"
        for d in [x for x in os.listdir(scenePath) if os.path.isdir(scenePath + x)]:
            if scene+".ttt" in os.listdir(scenePath + d):
                modeName = d
                scenePath += d + "/" + scene + ".ttt"
                break
        # start V-REP
        lock = fasteners.InterProcessLock(os.path.abspath(os.path.dirname(__file__)) + "lockfile")
        lock.acquire()
        self.IS_BOOT = is_boot
        if self.IS_BOOT:
            # change port number
            content = ""
            with open(self.VREP_DIR + exeDir + "remoteApiConnections.txt", "r") as f_handle:
                content = f_handle.read().splitlines()
            target = content[11].split("=")
            target[1] = " " + str(port)
            content[11] = "=".join(target)
            with open(self.VREP_DIR + exeDir + "remoteApiConnections.txt", "w") as f_handle:
                f_handle.write("\n".join(content))
            # open vrep
            vrepArgs = [self.VREP_DIR + exeDir + vrepExe, scenePath]
            if not is_render:
                vrepArgs.extend(["-h"])
            vrepArgs.extend(["&"])
            self.vrepProcess = subprocess.Popen(vrepArgs, stdout=open(os.devnull, 'w'), stderr=subprocess.STDOUT, preexec_fn=os.setsid)
            print("Enviornment was opened:\n{}\n{}".format(vrepArgs[0], scenePath))
        else:
            # get port number (assume that the opened vrep loaded the current text)
            content = ""
            with open(self.VREP_DIR + exeDir + "remoteApiConnections.txt", "r") as f_handle:
                content = f_handle.read().splitlines()
            target = content[11].split("=")
            port = int(target[1])
        # connect to V-REP
        ipAddress = "127.0.0.1"
        self.__ID = vrep.simxStart(ipAddress, port, True, True, 5000, 1)
        while self.__ID == -1:
            self.__ID = vrep.simxStart(ipAddress, port, True, True, 5000, 1)
        print("Connection succeeded: {}:{}".format(ipAddress, port))
        lock.release()
        # open scene if already booted
        if not self.IS_BOOT:
            vrep.simxLoadScene(self.__ID, scenePath, 0, vrep.simx_opmode_blocking)
            print("Scene was opened:\n{}".format(scenePath))
        # start to set constants
        vrep.simxSynchronous(self.__ID, True)
        vrep.simxStartSimulation(self.__ID, vrep.simx_opmode_blocking)
        vrep.simxSynchronousTrigger(self.__ID)
        if is_render:
            vrep.simxSetBooleanParameter(self.__ID, vrep.sim_boolparam_display_enabled, False, vrep.simx_opmode_oneshot)
        # set functions for corresponding modes
        if "normal" in modeName:
            self.__MODE = ModeN(self.__ID)
            self.DT , self.observation_space , self.action_space = self.__MODE.define()
        elif "multi_objective" in modeName:
            self.__MODE = ModeMO(self.__ID)
            self.DT , self.observation_space , self.action_space , self.TASK_NUM = self.__MODE.define()
        elif "multi_agent" in modeName:
            self.__MODE = ModeMA(self.__ID)
            self.DT , self.observation_space , self.action_space , self.AGE_NUM = self.__MODE.define()
        # stop simulation
        self.__stop()
        self.IS_RECORD = False



    def close(self):
        self.__stop()
        if self.IS_RECORD:
            self.__move()
        if self.IS_BOOT:
            vrep.simxFinish(self.__ID)
            os.killpg(os.getpgid(self.vrepProcess.pid), signal.SIGTERM)
            self.vrepProcess.wait()
            print("Enviornment was closed")
        else:
            vrep.simxCloseScene(self.__ID, vrep.simx_opmode_blocking)
            vrep.simxFinish(self.__ID)
            print("Scene was closed")
        time.sleep(2)   # just in case



    def reset(self):
        self.__stop()
        if self.IS_RECORD:
            self.__move()
            vrep.simxSetBooleanParameter(self.__ID, vrep.sim_boolparam_video_recording_triggered, True, vrep.simx_opmode_oneshot)
        vrep.simxSynchronous(self.__ID, True)
        vrep.simxStartSimulation(self.__ID, vrep.simx_opmode_blocking)
        self.__MODE.set(None)
        vrep.simxSynchronousTrigger(self.__ID)
        vrep.simxGetPingTime(self.__ID)
        # get initial states
        state , reward , done , info = self.__MODE.get()
        return state



    def step(self, action):
        # set actions
        self.__MODE.set(action)
        vrep.simxSynchronousTrigger(self.__ID)
        vrep.simxGetPingTime(self.__ID)
        # get new states
        return self.__MODE.get()



    def monitor(self, save_dir="./video", force=False):
        self.IS_RECORD = True
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        if force:
            self.videoName = save_dir + "/recording"
        else:
            self.videoName = save_dir + "/"



    def __stop(self):
        # vrep.simxSynchronous(self.__ID, False)
        vrep.simxStopSimulation(self.__ID, vrep.simx_opmode_blocking)
        while vrep.simxGetInMessageInfo(self.__ID, vrep.simx_headeroffset_server_state)[1] % 2 == 1:
            pass

    def __move(self):
        for f in os.listdir(self.VREP_DIR):
            if "recording_" in f:
                if self.videoName[-1] == "/":
                    shutil.move(self.VREP_DIR + f, self.videoName + f)
                else:
                    name , ext = os.path.splitext(f)
                    shutil.move(self.VREP_DIR + f, self.videoName + ext)

######################################################
# meta model
######################################################
class Mode(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, id):
        self.id_ = id
        self.dt_ = vrep.simxGetFloatSignal(self.id_, "dt", vrep.simx_opmode_blocking)[1]

    @abstractmethod
    def define(self):
        return [self.dt_ , self.s_obs_ , self.s_act_]

    @abstractmethod
    def set(self, action):
        pass

    @abstractmethod
    def get(self):
        self.done = self._check_Done()
        return self.state , self.reward , self.done , {}

    def _get_Space(self, soa, prefix="", dtype=np.float32):
        if "state" in soa:
            max_v = np.array(vrep.simxUnpackFloats(vrep.simxGetStringSignal(self.id_, prefix + "max_state", vrep.simx_opmode_blocking)[1]))
            min_v = np.array(vrep.simxUnpackFloats(vrep.simxGetStringSignal(self.id_, prefix + "min_state", vrep.simx_opmode_blocking)[1]))
        elif "action" in soa:
            max_v = np.array(vrep.simxUnpackFloats(vrep.simxGetStringSignal(self.id_, prefix + "max_action", vrep.simx_opmode_blocking)[1]))
            min_v = np.array(vrep.simxUnpackFloats(vrep.simxGetStringSignal(self.id_, prefix + "min_action", vrep.simx_opmode_blocking)[1]))
        return spaces.Box(min_v, max_v, dtype=dtype)

    def _get_StateReward(self, prefix="", init=False):
        v_opmode = vrep.simx_opmode_streaming if init else vrep.simx_opmode_buffer
        state = np.array( vrep.simxUnpackFloats( vrep.simxGetStringSignal(self.id_, prefix + "states", v_opmode)[1] ) )
        reward = vrep.simxGetFloatSignal(self.id_, prefix + "reward", v_opmode)[1]
        return state , reward

    def _check_Done(self, init=False):
        v_opmode = vrep.simx_opmode_streaming if init else vrep.simx_opmode_buffer
        return bool( vrep.simxGetIntegerSignal(self.id_, "done", v_opmode)[1] )

    def _set_Action(self, action, prefix=""):
        vrep.simxSetStringSignal(self.id_, prefix + "actions", vrep.simxPackFloats(action), vrep.simx_opmode_oneshot)

######################################################
# for normal scenes
######################################################
class ModeN(Mode):
    def __init__(self, id):
        super().__init__(id)
        self.s_obs_ = self._get_Space("state")
        self.s_act_ = self._get_Space("action")
        # variables will be received
        self.state = np.zeros(len(self.s_obs_.high))
        self.reward = 0.0
        self.done = False
        # variables will be sended
        self.action = np.zeros(len(self.s_act_.high))
        # enable streaming
        self.state , self.reward = self._get_StateReward(init=True)
        self.done = self._check_Done(init=True)

    def define(self):
        rtv = super().define()
        return rtv

    def set(self, action):
        if action is None:
            self.action = np.zeros_like(self.action)
        else:
            self.action = np.clip(action, self.s_act_.low, self.s_act_.high)
        self._set_Action(self.action)

    def get(self):
        self.state , self.reward = self._get_StateReward()
        return super().get()

######################################################
# for multi_objective scenes
######################################################
class ModeMO(Mode):
    def __init__(self, id):
        super().__init__(id)
        # constant shared with lua
        self.n_task = vrep.simxGetIntegerSignal(self.id_, "tasks", vrep.simx_opmode_blocking)[1]
        self.s_obs_ = self._get_Space("state")
        self.s_act_ = self._get_Space("action")
        # variables will be received
        self.state = np.zeros(len(self.s_obs_.high))
        self.reward = 0.0
        self.done = False
        # variables will be sended
        self.action = np.zeros(len(self.s_act_.high) + self.n_task)
        # enable streaming
        self.state , self.reward = self._get_StateReward(init=True)
        self.done = self._check_Done(init=True)

    def define(self):
        rtv = super().define()
        return rtv + [self.n_task]

    def set(self, action):
        if action is None:
            self.action = np.zeros_like(self.action)
        else:
            self.action[:-self.n_task] = np.clip(action[:-self.n_task], self.s_act_.low, self.s_act_.high)
            self.action[-self.n_task:] = action[-self.n_task:]
        self._set_Action(self.action)

    def get(self):
        self.state , self.reward = self._get_StateReward()
        return super().get()

######################################################
# for multi_agent scenes
######################################################
class ModeMA(Mode):
    def __init__(self, id):
        super().__init__(id)
        # constant shared with lua
        self.n_agent = vrep.simxGetIntegerSignal(self.id_, "agents", vrep.simx_opmode_blocking)[1]
        self.s_act_ = []
        self.s_obs_ = []
        for i in range(self.n_agent):
            self.s_obs_.append(self._get_Space("state", "Agent" + str(i+1) + "_"))
            self.s_act_.append(self._get_Space("action", "Agent" + str(i+1) + "_"))
        # variables will be received
        self.state = [None for _ in range(self.n_agent)]
        self.reward = np.zeros(self.n_agent)
        self.done = False
        # variables will be sended
        self.action = [np.zeros_like(self.s_act_[i].high) for i in range(self.n_agent)]
        # enable streaming
        for i in range(self.n_agent):
            self.state[i] , self.reward[i] = self._get_StateReward(prefix="Agent" + str(i+1) + "_", init=True)
        self.done = self._check_Done(init=True)

    def define(self):
        rtv = super().define()
        return rtv + [self.n_agent]

    def set(self, action):
        if action is None:
            for i in range(self.n_agent):
                self.action[i] = np.zeros_like(self.action[i])
                self._set_Action(self.action[i], "Agent" + str(i+1) + "_")
        else:
            for i in range(self.n_agent):
                self.action[i] = np.clip(action[i], self.s_act_[i].low, self.s_act_[i].high)
                self._set_Action(self.action[i], "Agent" + str(i+1) + "_")

    def get(self):
        for i in range(self.n_agent):
            self.state[i] , self.reward[i] = self._get_StateReward(prefix="Agent" + str(i+1) + "_")
        return super().get()
