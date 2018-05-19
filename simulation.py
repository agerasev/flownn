import numpy as np
import pyopencl as cl


class _BufferProto:
    def __init__(self, ctx, nparray, rw=cl.mem_flags.READ_WRITE, dual=False, **kwargs):
        self.ctx = ctx
        self.rw = rw
        self.dual = dual
        self.host = nparray
        
        self.buf = self._mkbuf(**kwargs)
        if dual:
            self.dbuf = self._mkbuf(**kwargs)
        
    def swap(self):
        if self.dual:
            self.buf, self.dbuf = self.dbuf, self.buf

    def load(self, queue):
        cl.enqueue_copy(queue, self.host, self.buf, **self._lkws())

class Buffer(_BufferProto):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _mkbuf(self):
        return cl.Buffer(self.ctx, self.rw | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.host)
    
    def _lkws(self):
        return {}
    
class Image2D(_BufferProto):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _mkbuf(self, fmt):
        return cl.Image(
            self.ctx, 
            self.rw | cl.mem_flags.COPY_HOST_PTR, 
            fmt, 
            hostbuf=self.host,
            shape=(self.host.shape[1], self.host.shape[0]),
        )
    
    def _lkws(self):
        return {
            "origin": (0, 0),
            "region": (self.host.shape[1], self.host.shape[0]),
        }


class Scene:
    def __init__(self, ctx, size, params):
        self.context = ctx
        self.size = size
        self.shape = (size[1], size[0])
        
        self.params = params
        
        self.queue = cl.CommandQueue(ctx)
        self.programs = {}
        for pn in ["draw", "render", "simulate"]:
            with open("%s.cl" % pn, "r") as f:
                self.programs[pn] = cl.Program(self.context, f.read()).build()
        
        self.buffers = {}
        
        self.buffers["color"] = Image2D(
            self.context,
            np.zeros((*self.shape, 4), dtype=np.float32),
            fmt=cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT),
            dual=True
        )
        self.buffers["velocity"] = Image2D(
            self.context,
            np.zeros((*self.shape, 2), dtype=np.float32),
            fmt=cl.ImageFormat(cl.channel_order.RG, cl.channel_type.FLOAT),
            dual=True
        )
        self.buffers["pressure"] = Image2D(
            self.context,
            np.zeros((*self.shape, 1), dtype=np.float32),
            fmt=cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT),
            dual=True
        )
        
        self.buffers["screen"] = Image2D(
            self.context,
            np.zeros((*self.shape, 4), dtype=np.uint8), 
            rw=cl.mem_flags.WRITE_ONLY, 
            fmt=cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8),
        )
        
        self.buffers["C"] = Buffer(
            self.context,
            np.array([
                self.params["time_step"],
                self.params["grid_size"],
                self.params["viscosity"],
            ], dtype=np.float32)
        )
        
    def step(self):
        prog = self.programs["simulate"]
        
        for i in range(self.params["pressure_steps"]):
            prog.pressure_compute_step(
                self.queue,
                self.size,
                None,
                self.buffers["C"].buf,
                self.buffers["velocity"].buf,
                self.buffers["pressure"].buf,
                self.buffers["pressure"].dbuf,
            )
            self.buffers["pressure"].swap()
        
        prog.pressure_subtract_gradient(
            self.queue,
            self.size,
            None,
            self.buffers["C"].buf,
            self.buffers["velocity"].buf,
            self.buffers["velocity"].dbuf,
            self.buffers["pressure"].buf,
            self.buffers["pressure"].dbuf,
        )
        self.buffers["velocity"].swap()
        self.buffers["pressure"].swap()
        
        prog.advect(
            self.queue,
            self.size,
            None,
            self.buffers["C"].buf,
            self.buffers["color"].buf,
            self.buffers["color"].dbuf,
            self.buffers["velocity"].buf,
            self.buffers["velocity"].dbuf,
        )
        self.buffers["color"].swap()
        self.buffers["velocity"].swap()
        
        prog.diffuse(
            self.queue,
            self.size,
            None,
            self.buffers["C"].buf,
            self.buffers["velocity"].buf,
            self.buffers["velocity"].dbuf,
        )
        self.buffers["velocity"].swap()
        
    def render(self):
        self.programs["render"].render(
            self.queue,
            self.size,
            None,
            self.buffers["color"].buf,
            self.buffers["velocity"].buf,
            self.buffers["pressure"].buf,
            self.buffers["screen"].buf,
        )
        self.buffers["screen"].load(self.queue)
        return self.buffers["screen"].host
    
    def draw(self, prim, col, vel, *args):
        col = np.array(list(col) + [1], dtype=np.float32)
        vel = np.array(list(vel) + [0,1], dtype=np.float32)
        if prim == "circle":
            pos = np.array(args[0], dtype=np.float32)
            rad = np.array([args[1]], dtype=np.float32)
            self.programs["draw"].circle(
                self.queue,
                self.size,
                None,
                self.buffers["C"].buf,
                self.buffers["color"].buf,
                self.buffers["velocity"].buf,
                pos, rad, col, vel
            )