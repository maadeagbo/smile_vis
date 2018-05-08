do
	matrix = require "scripts.matrix"

  data_manager = { data = {} }

  bounds_min = {0, 0}
  bounds_max = {0, 0}

  time_tracker = 0.0
  fps = 1.0/20.0
  once = true

  ideal_lat_iris_pos = { 0.100, 0.900 }
  ideal_lat_iris_dist = 0.05

  function set_bounds( point ) 
    -- set min
    bounds_min[1] = (point.x < bounds_min[1]) and point.x or bounds_min[1]
    bounds_min[2] = (point.y < bounds_min[2]) and point.y or bounds_min[2]
    -- set max
    bounds_max[1] = (point.x > bounds_max[1]) and point.x or bounds_max[1]
    bounds_max[2] = (point.y > bounds_max[2]) and point.y or bounds_max[2]
  end

  -- update function
  function data_manager:update( event, args, num_args )
    -- get data (if something is opened)
    if self.data.num_frames > 0 then
      -- animate
      time_tracker = time_tracker + ddLib.ftime()
      if time_tracker >= fps then
        time_tracker = 0
        self.data.idx = (self.data.idx + 1) % self.data.num_frames
        --ddLib.print("Frame: ", self.data.idx)
      end

      if once then
        -- get data of input
        p_in = { SController.get_input_data() }
        -- get data of ground truth
        p_gt = { SController.get_ground_data() }
        -- get neural net data
        p_calc = { SController.get_calc_data() }
        -- adjust camera parameters (orthographic projection)
        bounds_min = {p_in[1].x, p_in[1].y}
        bounds_max = {p_in[1].x, p_in[1].y}

        for i=1,#p_in do
          set_bounds(p_in[i])
        end

        for i=1,#p_gt do
          set_bounds(p_gt[i])
        end

        for i=1,#p_calc do
          set_bounds(p_calc[i])
        end

        -- add a bit of an offset
        self.data.ortho = {
          bounds_min[1] - 5.0, 
          bounds_max[1] + 2.0, 
          bounds_max[2] + 2.0, 
          bounds_min[2] - 2.0
        }
        self.data.tile = 0.05
        ddLib.print("Bounds: ", bounds_min[1], ", ", bounds_min[2], ", ", bounds_max[1], ", ", bounds_max[2])
        once = false
      end
      
      -- ddLib.print(bounds_min[1] - 100, ",", 
      --   bounds_max[1] + 100, ",", 
      --   bounds_max[2] + 100, ",", 
      --   bounds_min[2] - 100
      -- )
    end
  end
  
  return data_manager
end