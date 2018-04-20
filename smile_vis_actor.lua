-- Actor script for smile_vis project

do
	matrix = require "scripts.matrix"

	smile_vis_actor = {}

	speed = 5.0
	pitch = 0
	yaw = 0

	function smile_vis_actor:new( params )
		params = params or {}
		self.__index = self
		setmetatable(params, self)

		return params
	end

	function smile_vis_actor:update( event, args, num_args )
		assets = smile_vis_assets
		input = ddInput

		-- Get current position
		curr_pos = assets.nav_agent:pos()

		-- Get current facing direction and convert to vector
		curr_d = assets.cam_01:dir()
		v3_fdir = matrix{ curr_d.x, curr_d.y, curr_d.z }

		-- Direction facing to the right
		v3_udir = matrix{0, 1, 0}
		v3_rdir = matrix.cross( v3_fdir, v3_udir )

		mloc_x = input.mouse_x / assets.scr_x
		mloc_y = input.mouse_y / assets.scr_y

		-- Get frame time and setup new position variable
		new_pos = matrix{curr_pos.x, curr_pos.y, curr_pos.z}
		ftime = ddLib.ftime()

		-- rotation
		if input.mouse_b_l and not ddLib.mouse_over_UI() and mloc_x > 0.4 then

			--assets.cam_01:set_eulerPYR(pitch, 0, 0)
			--assets.nav_agent:set_eulerPYR(0, yaw, 0)

			new_pos[1][1] = new_pos[1][1] + input.mouse_x_delta * ftime 
			new_pos[2][1] = new_pos[2][1] + input.mouse_y_delta * -ftime

			assets.nav_agent:set_pos(new_pos[1][1], new_pos[2][1], new_pos[3][1])
		end
	end

	return smile_vis_actor
end
