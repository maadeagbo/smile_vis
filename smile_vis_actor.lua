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

		-- Get current position
		curr_pos = assets.nav_agent:pos()

		-- Get current facing direction and convert to vector
		curr_d = assets.cam_01:dir()
		v3_fdir = matrix{ curr_d.x, curr_d.y, curr_d.z }

		-- Direction facing to the right
		v3_udir = matrix{0, 1, 0}
		v3_rdir = matrix.cross( v3_fdir, v3_udir )

		--ddLib.print( string.format("Pos = %.3f, %.3f, %.3f",
			--curr_pos.x, curr_pos.y, curr_pos.z))

		-- Get frame time and setup new position variable
		new_pos = matrix{curr_pos.x, curr_pos.y, curr_pos.z}
		ftime = ddLib.ftime()

		-- left
		if ddInput.a then new_pos = new_pos - (v3_rdir * ftime * speed) end
		-- right
		if ddInput.d then new_pos = new_pos + (v3_rdir * ftime * speed) end
		-- forward
		if ddInput.w then new_pos = new_pos + (v3_fdir * ftime * speed) end
		-- back
		if ddInput.s then new_pos = new_pos - (v3_fdir * ftime * speed) end
		-- down
		if ddInput.l_shift then new_pos = new_pos - (v3_udir * ftime * speed) end
		-- up
		if ddInput.space then new_pos = new_pos + (v3_udir * ftime * speed) end

		-- rotation
		if ddInput.mouse_b_l then
			pitch = pitch + ddInput.mouse_y_delta * 1.0/speed
			yaw = yaw + ddInput.mouse_x_delta * 1.0/speed

			assets.cam_01:set_eulerPYR(pitch, 0, 0)
			assets.nav_agent:set_eulerPYR(0, yaw, 0)
		end
		assets.nav_agent:set_pos(new_pos[1][1], new_pos[2][1], new_pos[3][1])
	end

	return smile_vis_actor
end
