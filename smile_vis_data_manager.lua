do
	matrix = require "scripts.matrix"

  data_manager = { data = {} }

  bounds_min = {0, 0}
  bounds_max = {0, 0}

  time_tracker = 0.0
  fps = 1.0/20.0

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
    -- calculate offset 
    if event == "calc_offset" then
      if self.data.num_frames <= 0 then
        ddLib.print("No smile file loaded")
      else
        p_in = { SController.get_input_data() } -- input
        p_gt = { SController.get_ground_data() } -- ground truth
        p_calc = { SController.get_calc_data() } -- output

        -- get translation offset
        t_vec = p_in[3]
        t_vec = { -t_vec.x, -t_vec.y }
        ddLib.print("Translation vector(", #p_calc, "): ", t_vec[1], ",", t_vec[2])
        
        -- apply delta translation to all points
        for i=1,#p_in do
          ddLib.print("  #", i, " = ", p_in[i].x, ", ", p_in[i].y)
          p_in[i].x = p_in[i].x + t_vec[1]
          p_in[i].y = p_in[i].y + t_vec[2]
          ddLib.print("  --->", p_in[i].x, ", ", p_in[i].y)
        end

        -- get rotation offset b/t lateral & medial iris
        -- math.atan2(l_iris_y, l_iris_x)

        -- apply negative rotation to all points (at the current pos)

        -- scale points so that iris distance is set to a canonical distance

        -- apply translation to all points to move iris to canonical position
      end
    end

    -- get data (if something is opened)
    if self.data.num_frames > 0 then
      -- animate
      time_tracker = time_tracker + ddLib.ftime()
      if time_tracker >= fps then
        time_tracker = 0
        self.data.idx = (self.data.idx + 1) % self.data.num_frames
        --ddLib.print("Frame: ", self.data.idx)
      end

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
        bounds_min[1] - 150, 
        bounds_max[1] + 50, 
        bounds_max[2] + 50, 
        bounds_min[2] - 50
      }
      -- ddLib.print(bounds_min[1] - 100, ",", 
      --   bounds_max[1] + 100, ",", 
      --   bounds_max[2] + 100, ",", 
      --   bounds_min[2] - 100
      -- )
    end
  end
  
  return data_manager
end