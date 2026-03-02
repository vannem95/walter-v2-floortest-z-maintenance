#include "operational-space-control/walter_sr/osc_node.h"

// Your anonymous namespace with Casadi functions goes here.
namespace {
    FunctionOperations Aeq_ops{
        .incref=Aeq_incref, .checkout=Aeq_checkout, .eval=Aeq, .release=Aeq_release, .decref=Aeq_decref};
    FunctionOperations beq_ops{
        .incref=beq_incref, .checkout=beq_checkout, .eval=beq, .release=beq_release, .decref=beq_decref};
    FunctionOperations Aineq_ops{
        .incref=Aineq_incref, .checkout=Aineq_checkout, .eval=Aineq, .release=Aineq_release, .decref=Aineq_decref};
    FunctionOperations bineq_ops{
        .incref=bineq_incref, .checkout=bineq_checkout, .eval=bineq, .release=bineq_release, .decref=bineq_decref};
    FunctionOperations H_ops{
        .incref=H_incref, .checkout=H_checkout, .eval=H, .release=H_release, .decref=H_decref};
    FunctionOperations f_ops{
        .incref=f_incref, .checkout=f_checkout, .eval=f, .release=f_release, .decref=f_decref};
    using AeqParams = FunctionParams<Aeq_SZ_ARG, Aeq_SZ_RES, Aeq_SZ_IW, Aeq_SZ_W, optimization::Aeq_rows, optimization::Aeq_cols, optimization::Aeq_sz, 4>;
    using beqParams = FunctionParams<beq_SZ_ARG, beq_SZ_RES, beq_SZ_IW, beq_SZ_W, optimization::beq_sz, 1, optimization::beq_sz, 4>;
    using AineqParams = FunctionParams<Aineq_SZ_ARG, Aineq_SZ_RES, Aineq_SZ_IW, Aineq_SZ_W, optimization::Aineq_rows, optimization::Aineq_cols, optimization::Aineq_sz, 1>;
    using bineqParams = FunctionParams<bineq_SZ_ARG, bineq_SZ_RES, bineq_SZ_IW, bineq_SZ_W, optimization::bineq_sz, 1, optimization::bineq_sz, 1>;
    using HParams = FunctionParams<H_SZ_ARG, H_SZ_RES, H_SZ_IW, H_SZ_W, optimization::H_rows, optimization::H_cols, optimization::H_sz, 4>;
    using fParams = FunctionParams<f_SZ_ARG, f_SZ_RES, f_SZ_IW, f_SZ_W, optimization::f_sz, 1, optimization::f_sz, 4>;

    // Helper function definitions
    template <typename T>
    bool contains(const std::vector<T>& vec, const T& value) {
        return std::find(vec.begin(), vec.end(), value) != vec.end();
    }
    
    std::vector<int> getSiteIdsOnSameBodyAsGeom(const mjModel* m, int geom_id) {
        std::vector<int> associated_site_ids;
        if (geom_id < 0 || geom_id >= m->ngeom) {
            std::cerr << "Error: Invalid geom ID: " << geom_id << std::endl;
            return associated_site_ids;
        }
        int geom_body_id = m->geom_bodyid[geom_id];
        for (int i = 0; i < m->nsite; ++i) {
            if (m->site_bodyid[i] == geom_body_id) {
                associated_site_ids.push_back(i);
            }
        }
        return associated_site_ids;
    }
    
    std::vector<int> getBinaryRepresentation_std_find(const std::vector<int>& A, const std::vector<int>& B) {
        std::vector<int> C;
        C.reserve(B.size());
        for (int b_element : B) {
            auto it = std::find(A.begin(), A.end(), b_element);
            C.push_back((it != A.end()) ? 1 : 0);
        }
        return C;
    }
}

// Full constructor implementation
OSCNode::OSCNode(const std::string& xml_path)
    : Node("osc_node"),
      xml_path_(xml_path),
      solution_(Vector<optimization::design_vector_size>::Zero()),
      dual_solution_(Vector<optimization::constraint_matrix_rows>::Zero()),
      design_vector_(Vector<optimization::design_vector_size>::Zero()),
      infinity_(OSQP_INFTY),
      big_number_(1e4),
      Abox_(MatrixColMajor<optimization::design_vector_size, optimization::design_vector_size>::Identity()),
      dv_lb_(Vector<optimization::dv_size>::Constant(-infinity_)),
      dv_ub_(Vector<optimization::dv_size>::Constant(infinity_)),
      u_lb_({-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0}),
      u_ub_({10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0}),
      z_lb_({
          -infinity_, -infinity_, 0.0, -infinity_, -infinity_, 0.0, -infinity_, -infinity_, 0.0, -infinity_, -infinity_, 0.0,
          -infinity_, -infinity_, 0.0, -infinity_, -infinity_, 0.0, -infinity_, -infinity_, 0.0, -infinity_, -infinity_, 0.0}),
      z_ub_({
          infinity_, infinity_, big_number_, infinity_, infinity_, big_number_, infinity_, infinity_, big_number_, infinity_, infinity_, big_number_,
          infinity_, infinity_, big_number_, infinity_, infinity_, big_number_, infinity_, infinity_, big_number_, infinity_, infinity_, big_number_}),
      bineq_lb_(Vector<optimization::bineq_sz>::Constant(-infinity_))

{
    // --- Mujoco initialization ---
    char error[1000];
    mj_model_ = mj_loadXML(xml_path_.c_str(), nullptr, error, 1000);
    if (!mj_model_) {
        RCLCPP_FATAL(this->get_logger(), "Failed to load Mujoco Model: %s", error);
        throw std::runtime_error("Failed to load Mujoco Model.");
    }
    mj_model_->opt.timestep = 0.002;
    mj_data_ = mj_makeData(mj_model_);

    mj_resetDataKeyframe(mj_model_, mj_data_, 5); // 
    mj_forward(mj_model_, mj_data_); // Compute initial kinematics
    
    
    // Thighs: 0, 2, 4, 6 in the 8-DOF motor array.
    // They correspond to indices 7, 9, 11, 13 in the full mj_data_->qpos array.
    // initial_tlh_angular_position_ = mj_data_->qpos[7 + 0]; // Index 7 (Motor 0)
    initial_tlh_angular_position_ = mj_data_->qpos[7 + 0]; // Index 7 (Motor 0)
    initial_trh_angular_position_ = mj_data_->qpos[7 + 2]; // Index 9 (Motor 2)
    initial_hlh_angular_position_ = mj_data_->qpos[7 + 4]; // Index 11 (Motor 4)
    initial_hrh_angular_position_ = mj_data_->qpos[7 + 6]; // Index 13 (Motor 6)

    // Shins: 1, 3, 5, 7 in the 8-DOF motor array.
    // They correspond to indices 8, 10, 12, 14 in the full mj_data_->qpos array.
    initial_tl_angular_position_ = mj_data_->qpos[7 + 1]; // Index 8 (Motor 1)
    initial_tr_angular_position_ = mj_data_->qpos[7 + 3]; // Index 10 (Motor 3)
    initial_hl_angular_position_ = mj_data_->qpos[7 + 5]; // Index 12 (Motor 5)
    initial_hr_angular_position_ = mj_data_->qpos[7 + 7]; // Index 14 (Motor 7)

    
    
    
    // Populate the site and body ID vectors
    for (const std::string_view& site : model::site_list) {
        std::string site_str = std::string(site);
        int id = mj_name2id(mj_model_, mjOBJ_SITE, site_str.data());
        assert(id != -1 && "Site not found in model.");
        sites_.push_back(site_str);
        site_ids_.push_back(id);
    }
    for (const std::string_view& site : model::noncontact_site_list) {
        std::string site_str = std::string(site);
        int id = mj_name2id(mj_model_, mjOBJ_SITE, site_str.data());
        assert(id != -1 && "Site not found in model.");
        noncontact_sites_.push_back(site_str);
        noncontact_site_ids_.push_back(id);
    }
    for (const std::string_view& site : model::contact_site_list) {
        std::string site_str = std::string(site);
        int id = mj_name2id(mj_model_, mjOBJ_SITE, site_str.data());
        assert(id != -1 && "Site not found in model.");
        contact_sites_.push_back(site_str);
        contact_site_ids_.push_back(id);
    }
    for (const std::string_view& body : model::body_list) {
        std::string body_str = std::string(body);
        int id = mj_name2id(mj_model_, mjOBJ_BODY, body_str.data());
        assert(id != -1 && "Body not found in model.");
        bodies_.push_back(body_str);
        body_ids_.push_back(id);
    }

    // SAFETY CHECK: Verify Site-Body Alignment
    for (size_t i = 0; i < site_ids_.size(); i++) {
        int site_id = site_ids_[i];
        int body_id_from_list = body_ids_[i]; // The one from your generated list
        int body_id_real = mj_model_->site_bodyid[site_id]; // The truth from MuJoCo

        if (body_id_from_list != body_id_real) {
            RCLCPP_FATAL(this->get_logger(), 
                "MISMATCH at index %zu! Site '%s' is on Body %d, but List says Body %d", 
                i, sites_[i].c_str(), body_id_real, body_id_from_list);
            throw std::runtime_error("Kinematic Chain Mismatch");
        }
    }
    
    assert(site_ids_.size() == body_ids_.size() && "Number of Sites and Bodies must be equal.");

    // --- Optimization Initialization ---
    // Create an initial state message to use for setup.
    OSCMujocoState initial_state_msg;
    // initial_state_msg.motor_position.assign(model::nu_size, 0.0);
    // initial_state_msg.motor_velocity.assign(model::nu_size, 0.0);
    // initial_state_msg.torque_estimate.assign(model::nu_size, 0.0);
    // initial_state_msg.body_rotation.assign(4, 0.0);
    // initial_state_msg.linear_body_velocity.assign(3, 0.0);
    // initial_state_msg.angular_body_velocity.assign(3, 0.0);
    // initial_state_msg.contact_mask.assign(model::contact_site_ids_size, 0.0);

    std::fill(initial_state_msg.motor_position.begin(), initial_state_msg.motor_position.end(), 0.0f);
    std::fill(initial_state_msg.motor_velocity.begin(), initial_state_msg.motor_velocity.end(), 0.0f);
    std::fill(initial_state_msg.torque_estimate.begin(), initial_state_msg.torque_estimate.end(), 0.0f);
    std::fill(initial_state_msg.body_rotation.begin(), initial_state_msg.body_rotation.end(), 0.0f);
    std::fill(initial_state_msg.linear_body_velocity.begin(), initial_state_msg.linear_body_velocity.end(), 0.0f);
    std::fill(initial_state_msg.angular_body_velocity.begin(), initial_state_msg.angular_body_velocity.end(), 0.0f);
    std::fill(initial_state_msg.contact_mask.begin(), initial_state_msg.contact_mask.end(), false);
    
    
    state_callback(std::make_shared<OSCMujocoState>(initial_state_msg));
    
    // update_mj_data();

    // --- Optimization Initialization ---
    // Instead of using a dummy ROS message to set state_ to zero, 
    // populate state_ with the actual initial Keyframe 5 data from mj_data_.
    
    // 1. Populate state_.motor_position from mj_data_->qpos
    //    Motor positions start at index 7 in the floating-base qpos array (3-pos + 4-quat).
    
    // Assuming model::nu_size is 8:
    // for (size_t i = 0; i < model::nu_size; ++i) {
    //     // qpos index = 7 (base pos/quat end) + i (motor index)
    //     state_.motor_position(i) = mj_data_->qpos[7 + i];
    //     // Ensure other essential fields are also non-zero if needed, 
    //     // e.g., base rotation:
    //     if (i < 4) {
    //         state_.body_rotation(i) = mj_data_->qpos[3 + i];
    //     }
    // }
    // // You can clear velocities and torques as they should start at zero.
    // state_.motor_velocity.setZero();
    // state_.linear_body_velocity.setZero();
    // state_.angular_body_velocity.setZero();
    // state_.torque_estimate.setZero();



    Vector<model::nq_size> qpos = Eigen::Map<Vector<model::nq_size>>(mj_data_->qpos);
    // initial_position_ = qpos(Eigen::seqN(0, 3));    
    
    Vector<model::contact_site_ids_size> initial_contact_mask = Vector<model::contact_site_ids_size>::Zero();

    absl::Status result = set_up_optimization(initial_contact_mask);
    if (!result.ok()) {
        RCLCPP_FATAL(this->get_logger(), "Failed to initialize optimization: %s", result.message().data());
        throw std::runtime_error("Failed to initialize optimization.");
    }    
    
    // --- ROS 2 communication setup ---
    state_subscriber_ = this->create_subscription<OSCMujocoState>(
        "/state_estimator/state", 1, std::bind(&OSCNode::state_callback, this, std::placeholders::_1));
    // taskspace_targets_subscriber_ = this->create_subscription<OSCTaskspaceTargets>(
    //     "osc/taskspace_targets", 10, std::bind(&OSCNode::taskspace_targets_callback, this, std::placeholders::_1));
    torque_publisher_ = this->create_publisher<Command>("walter/command", 1);
    // torque_publisher_ = this->create_publisher<OSCTorqueCommand>("walter/command", 10);
    // New: 5000 microseconds (5 ms = 200 Hz)
    timer_ = this->create_wall_timer(std::chrono::microseconds(5000), std::bind(&OSCNode::timer_callback, this));

    rclcpp::on_shutdown([this]() {
        RCLCPP_WARN(this->get_logger(), "Shutdown signal received. Attempting to stop robot...");
        this->stop_robot();
    });    
}

OSCNode::~OSCNode() {
    mj_deleteData(mj_data_);
    mj_deleteModel(mj_model_);
}

// ===============================================================================================================
// Full implementation of all methods
void OSCNode::state_callback(const OSCMujocoState::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    // Manually copy and cast each member to the correct double type
    for (size_t i = 0; i < model::nu_size; ++i) {
        state_.motor_position(i) = static_cast<double>(msg->motor_position[i]);
        state_.motor_velocity(i) = static_cast<double>(msg->motor_velocity[i]);
        state_.torque_estimate(i) = static_cast<double>(msg->torque_estimate[i]);

        // CAPTURE DETECTED POSITION
        last_detected_motor_position_(i) = state_.motor_position(i);        
        
    }

    // CAPTURE STATE READ TIME
    state_read_time_ = std::chrono::high_resolution_clock::now();    
    
    for (size_t i = 0; i < 4; ++i) {
        state_.body_rotation(i) = static_cast<double>(msg->body_rotation[i]);
    }

    for (size_t i = 0; i < 3; ++i) {
        state_.linear_body_velocity(i) = static_cast<double>(msg->linear_body_velocity[i]);
        state_.angular_body_velocity(i) = static_cast<double>(msg->angular_body_velocity[i]);
    }

    for (size_t i = 0; i < model::contact_site_ids_size; ++i) {
        state_.contact_mask(i) = static_cast<double>(msg->contact_mask[i]);
    }
    is_state_received_ = true;    
}



// ===============================================================================================================
void OSCNode::timer_callback() {
    // --- 1. DECLARE LOCAL COPIES ---
    State local_state; 
    bool local_safety_override_active;
    std::chrono::time_point<std::chrono::high_resolution_clock> local_state_read_time;
    
    double current_time = this->now().seconds();

    // --- DIAGNOSTIC START: Capture the exact start time ---
    auto t_start_execution = std::chrono::high_resolution_clock::now(); 
    // --- DIAGNOSTIC END ---

    { // --- CRITICAL SECTION START (Locked, FAST) ---
        std::lock_guard<std::mutex> lock_state(state_mutex_);
        
        // Copy the shared state members needed for the control loop
        local_state = state_; 
        local_safety_override_active = safety_override_active_;
        local_state_read_time = state_read_time_;
        
        // Check for state gating before proceeding
        if (!is_state_received_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Waiting for initial state message...");
            return; 
        }

    } // --- CRITICAL SECTION END (Lock released!) ---

    time_wait_for_execution_ms_ = std::chrono::duration<double, std::milli>(t_start_execution - local_state_read_time).count();    

    // Check for first call or zero time step
    if (last_time_ == 0.0) {
        update_mj_data(local_state); // Pass local state
        last_time_ = current_time;
        return; 
    }

    

    // --- 2. Mandatory Joint Limit Check (Outer Loop - UNLOCKED) ---
    // If local_safety_override_active is true, limit_hit remains true, and the check loop is skipped.
    bool limit_hit = local_safety_override_active; 
    
    if (!local_safety_override_active) {
        
        const double SHIN_LIMIT = M_PI / 2.0;
        const double THIGH_LIMIT = M_PI / 4.0;
        
        // Check Thighs (0, 2, 4, 6) using local_state.motor_position
        for (size_t i : {0, 2, 4, 6}) {
            if (std::abs(local_state.motor_position(i)) >= THIGH_LIMIT) {
                limit_hit = true;
                RCLCPP_WARN_ONCE(this->get_logger(), "Absolute THIGH limit (%.2f rad) hit on motor index %zu. Overriding control.", THIGH_LIMIT, i);
                break; 
            }
        }
        
        // Check Shins (1, 3, 5, 7)
        if (!limit_hit) {
            for (size_t i : {1, 3, 5, 7}) {
                if (std::abs(local_state.motor_position(i)) >= SHIN_LIMIT) {
                    limit_hit = true;
                    RCLCPP_WARN_ONCE(this->get_logger(), "Absolute SHIN limit (%.2f rad) hit on motor index %zu. Overriding control.", SHIN_LIMIT, i);
                    break; 
                }
            }
        }

        // If a new limit was hit, set the SHARED safety flag to true (permanently)
        if (limit_hit) { 
            std::lock_guard<std::mutex> lock_state(state_mutex_);
            safety_override_active_ = true;
            local_safety_override_active = true; // Update local for subsequent steps
        }
    }


    // --- 3. Conditional OSC Calculation and Solve (UNLOCKED) ---
    if (!local_safety_override_active) {

        // 1. Update Mujoco Data for Kinematics (using local_state)
        
        // --- TIMING POINT A: START MUJOCO/KINEMATICS ---
        auto t_start_kinematics = std::chrono::high_resolution_clock::now();        
        
        update_mj_data(local_state); 

        // 2b. Define Targets and Calculate DDQ Commands

        // Sim - on ground
        // shin - (kp - 80*100 — kd - 80*10) 200/20, 300/30
        double shin_factor = 1.0;
        double shin_kp = 400.0*shin_factor; double shin_kv = 20.0*shin_factor;

        // thigh - (kp - 10*100 — kd - 10*10) 200/20
        double thigh_factor = 1.0;
        double thigh_kp = 400.0*thigh_factor; double thigh_kv = 20.0*thigh_factor;

        double shin_pos_target = 0.0;
        double thigh_pos_target = 0.55; 

        double rot_vel_target = 0.0; 

        // Shin DDQ Commands (using local_state)
        double tl_ddq_cmd  = shin_kp * (0.0 + shin_pos_target - local_state.motor_position(1)) + shin_kv * (rot_vel_target - local_state.motor_velocity(1));
        double tr_ddq_cmd  = shin_kp * (0.0 + shin_pos_target - local_state.motor_position(3)) + shin_kv * (rot_vel_target - local_state.motor_velocity(3));
        double hl_ddq_cmd  = shin_kp * (0.0 - shin_pos_target - local_state.motor_position(5)) + shin_kv * (rot_vel_target - local_state.motor_velocity(5));
        double hr_ddq_cmd  = shin_kp * (0.0 - shin_pos_target - local_state.motor_position(7)) + shin_kv * (rot_vel_target - local_state.motor_velocity(7));

        // Thigh DDQ Commands (using local_state)
        double tlh_ddq_cmd = thigh_kp * (0.0 + thigh_pos_target - local_state.motor_position(0)) + thigh_kv * (rot_vel_target - local_state.motor_velocity(0));
        double trh_ddq_cmd = thigh_kp * (0.0 + thigh_pos_target - local_state.motor_position(2)) + thigh_kv * (rot_vel_target - local_state.motor_velocity(2));
        double hlh_ddq_cmd = thigh_kp * (0.0 - thigh_pos_target - local_state.motor_position(4)) + thigh_kv * (rot_vel_target - local_state.motor_velocity(4));
        double hrh_ddq_cmd = thigh_kp * (0.0 - thigh_pos_target - local_state.motor_position(6)) + thigh_kv * (rot_vel_target - local_state.motor_velocity(6));

        // Populate Taskspace Targets Matrix 
        taskspace_targets_.setZero(); 
        taskspace_targets_.row(1)(4) = tl_ddq_cmd; taskspace_targets_.row(2)(4) = tr_ddq_cmd;
        taskspace_targets_.row(3)(4) = hl_ddq_cmd; taskspace_targets_.row(4)(4) = hr_ddq_cmd;
        taskspace_targets_.row(5)(4) = tlh_ddq_cmd; taskspace_targets_.row(6)(4) = trh_ddq_cmd;
        taskspace_targets_.row(7)(4) = hlh_ddq_cmd; taskspace_targets_.row(8)(4) = hrh_ddq_cmd;

        
        // Solve Optimization
        update_osc_data();
        
        // --- TIMING POINT B: END MUJOCO/KINEMATICS, START CASADI/OSQP DATA ---        
        auto t_start_casadi = std::chrono::high_resolution_clock::now();        
        
        update_optimization_data();
        std::ignore = update_optimization(local_state.contact_mask); 

        // --- TIMING POINT C: END CASADI/OSQP DATA, START SOLVE ---
        auto t_start_solve = std::chrono::high_resolution_clock::now();


        
        // --- DEBUGGING BLOCK: PRINT SOLVER STATE ---
        // Print only once per second to avoid flooding console
        // static auto last_print_time = std::chrono::steady_clock::now();
        // auto now = std::chrono::steady_clock::now();
        // if (std::chrono::duration_cast<std::chrono::seconds>(now - last_print_time).count() >= 1) {
            // last_print_time = now;

        std::stringstream ss;
        ss << "\n--- OSC STATE DEBUG ---\n";
        
        // 1. Check if solver thinks feet are touching ground
        ss << "Contact Mask: [ ";
        for(int i=0; i<model::contact_site_ids_size; ++i) ss << local_state.contact_mask(i) << " ";
        ss << "]\n";

        // 2. Check estimated robot mass (from qM)
        // Trace of upper-left 3x3 of Mass Matrix roughly correlates to mass
        double approx_mass = osc_data_.mass_matrix(0,0); 
        ss << "Mass Matrix (0,0): " << approx_mass << " (Should be ~total robot mass)\n";

        // 3. Check qpos (Altitude)
        ss << "Torso x (qpos[0]): " << mj_data_->qpos[0] << "\n";
        ss << "Torso y (qpos[1]): " << mj_data_->qpos[1] << "\n";
        ss << "Torso Height (qpos[2]): " << mj_data_->qpos[2] << "\n";

        // 4. Check Orientation
        ss << "Torso Quat (w,x,y,z): " << mj_data_->qpos[3] << ", " << mj_data_->qpos[4] 
        << ", " << mj_data_->qpos[5] << ", " << mj_data_->qpos[6] << "\n";

        // 3. Joint Angles (Motors)
        // These start at index 7 for a Floating Base robot
        ss << "Motor Angles (Rad):  [ ";
        for (int i = 0; i < model::nu_size; ++i) {
            ss << mj_data_->qpos[7 + i] << " ";
        }
        ss << "]\n";

        // 4. Joint Velocities (qvel) - Optional but useful
        // Note: qvel is different. [0-2] Linear, [3-5] Angular, [6+] Joints
        ss << "Motor Velocities:    [ ";
        for (int i = 0; i < model::nu_size; ++i) {
            ss << mj_data_->qvel[6 + i] << " ";
        }
        ss << "]\n";      

        // 5. Check Gravity Vector (qfrc_bias)
        // The first 3 elements of qfrc_bias should be approx [0, 0, mass*9.81]
        ss << "Gravity/Bias Force (0-2): " 
        << osc_data_.coriolis_matrix(0) << ", " 
        << osc_data_.coriolis_matrix(1) << ", " 
        << osc_data_.coriolis_matrix(2) << "\n";

        RCLCPP_INFO(this->get_logger(), "%s", ss.str().c_str());
        // }
        // -------------------------------------------
        


        // OLD: solve_optimization();
        // NEW: Check result
        bool solver_success = solve_optimization();
        
        if (!solver_success) {
            // CRITICAL: Trigger safety override immediately if math fails
            std::lock_guard<std::mutex> lock(state_mutex_);
            safety_override_active_ = true;
            local_safety_override_active = true; 
            RCLCPP_ERROR(this->get_logger(), "Optimization failed! Engaging safety override.");
        }
                
        // --- TIMING POINT D: END SOLVE ---
        auto t_end_solve = std::chrono::high_resolution_clock::now();
        
        // Calculate and store internal execution times 
        time_mujoco_update_ms_ = std::chrono::duration<double, std::milli>(t_start_casadi - t_start_kinematics).count();
        time_casadi_update_ms_ = std::chrono::duration<double, std::milli>(t_start_solve - t_start_casadi).count();
        time_osqp_solve_ms_ = std::chrono::duration<double, std::milli>(t_end_solve - t_start_solve).count();                
    }

    if (!local_safety_override_active) {
        std::lock_guard<std::mutex> lock(state_mutex_); // Quick lock
        if (safety_override_active_) {
            // Aha! The flag changed from false to true while we were calculating.
            // Update our local copy so we send the safety command instead.
            local_safety_override_active = true; 
        }
    }    

    // Publish using the determined safety status and the captured timestamp
    publish_torque_command(local_safety_override_active, local_state_read_time); 
}
// ---------------------------------------------------------------------------------------------------------





// ===============================================================================================================
void OSCNode::update_mj_data(const State& current_state) {
    // =========================================================================
    // PART 1: PREPARE POSITION (QPOS)
    // =========================================================================
    
    // 1. Map Body Orientation (IMU) -> qpos[3-6]
    // MuJoCo Quaternion order: [w, x, y, z] - imu uses x y z w
    mj_data_->qpos[3] = current_state.body_rotation(0); // w
    mj_data_->qpos[4] = current_state.body_rotation(1); // x
    mj_data_->qpos[5] = current_state.body_rotation(2); // y
    mj_data_->qpos[6] = current_state.body_rotation(3); // z

    // Safety: Handle zero quaternion
    if (current_state.body_rotation.norm() < 1e-6) {
        mj_data_->qpos[3] = 1.0; 
    }

    // 2. Map Motor Positions -> qpos[7...]
    // Note: Adjust indices if your robot has fewer/more joints
    for (int i = 0; i < model::nu_size; ++i) {
        mj_data_->qpos[7 + i] = current_state.motor_position(i);
    }

    // 3. Reset Body Position to (0,0,0) for "Probing"
    mj_data_->qpos[0] = 0.0;
    mj_data_->qpos[1] = 0.0;
    mj_data_->qpos[2] = 0.0;

    // =========================================================================
    // PART 2: ANCHORING LOGIC (Find Body Position)
    // =========================================================================
    
    // Run Kinematics on the "Probed" state
    mj_fwdPosition(mj_model_, mj_data_);

    double sum_active_x = 0.0;
    double sum_active_y = 0.0;
    double lowest_foot_z = 100.0;
    int active_feet_count = 0;

    // A. Find absolute floor height
    for (int id : contact_site_ids_) { 
        double z = mj_data_->site_xpos[3 * id + 2];
        if (z < lowest_foot_z) lowest_foot_z = z;
    }

    // B. Average the active feet
    const double CONTACT_THRESHOLD = 0.01;
    for (int id : contact_site_ids_) { 
        double x = mj_data_->site_xpos[3 * id + 0];
        double y = mj_data_->site_xpos[3 * id + 1];
        double z = mj_data_->site_xpos[3 * id + 2];

        // If foot is near the floor, use it for support center
        if (std::abs(z - lowest_foot_z) <= CONTACT_THRESHOLD) {
            sum_active_x += x;
            sum_active_y += y;
            active_feet_count++;
        }
    }

    // C. Invert the vector (If feet are at X, Body must be at -X)
    double real_body_x = 0.0;
    double real_body_y = 0.0;
    
    if (active_feet_count > 0) {
        real_body_x = -(sum_active_x / active_feet_count);
        real_body_y = -(sum_active_y / active_feet_count);
    }

    double real_body_z = std::clamp(-lowest_foot_z, 0.10, 0.80); 

    // D. Apply calculated Body Position
    mj_data_->qpos[0] = real_body_x;
    mj_data_->qpos[1] = real_body_y;
    mj_data_->qpos[2] = real_body_z;

    // =========================================================================
    // PART 3: PREPARE VELOCITY (QVEL) - CRITICAL!
    // =========================================================================
    
    // 1. Linear Body Velocity (From estimator, or 0 if unknown)
    mj_data_->qvel[0] = current_state.linear_body_velocity(0);
    mj_data_->qvel[1] = current_state.linear_body_velocity(1);
    mj_data_->qvel[2] = current_state.linear_body_velocity(2);

    // 2. Angular Body Velocity (From IMU Gyro)
    mj_data_->qvel[3] = current_state.angular_body_velocity(0);
    mj_data_->qvel[4] = current_state.angular_body_velocity(1);
    mj_data_->qvel[5] = current_state.angular_body_velocity(2);

    // 3. Motor Velocities
    for (int i = 0; i < model::nu_size; ++i) {
        mj_data_->qvel[6 + i] = current_state.motor_velocity(i);
    }

    // =========================================================================
    // PART 4: FINAL COMPUTATION
    // =========================================================================
    
    // Update Jacobians (mj_jac) and Mass Matrix (qM) with the FULL correct state
    mj_fwdPosition(mj_model_, mj_data_);
    mj_fwdVelocity(mj_model_, mj_data_); 
    
    // Update site positions for your controller's use
    points_ = Eigen::Map<Matrix<model::site_ids_size, 3>>(
        mj_data_->site_xpos)(site_ids_, Eigen::placeholders::all);
}



// ===============================================================================================================
void OSCNode::update_osc_data() {
    Matrix<model::nv_size, model::nv_size> mass_matrix = Matrix<model::nv_size, model::nv_size>::Zero();
    mj_fullM(mj_model_, mass_matrix.data(), mj_data_->qM);
    Vector<model::nv_size> coriolis_matrix = Eigen::Map<Vector<model::nv_size>>(mj_data_->qfrc_bias);
    Vector<model::nq_size> generalized_positions = Eigen::Map<Vector<model::nq_size>>(mj_data_->qpos);
    Vector<model::nv_size> generalized_velocities = Eigen::Map<Vector<model::nv_size>>(mj_data_->qvel);

    Matrix<optimization::p_size, model::nv_size> jacobian_translation = Matrix<optimization::p_size, model::nv_size>::Zero();
    Matrix<optimization::r_size, model::nv_size> jacobian_rotation = Matrix<optimization::r_size, model::nv_size>::Zero();
    Matrix<optimization::p_size, model::nv_size> jacobian_dot_translation = Matrix<optimization::p_size, model::nv_size>::Zero();
    Matrix<optimization::r_size, model::nv_size> jacobian_dot_rotation = Matrix<optimization::r_size, model::nv_size>::Zero();
    
    for (int i = 0; i < model::body_ids_size; i++) {
        Matrix<3, model::nv_size> jacp = Matrix<3, model::nv_size>::Zero();
        Matrix<3, model::nv_size> jacr = Matrix<3, model::nv_size>::Zero();
        Matrix<3, model::nv_size> jacp_dot = Matrix<3, model::nv_size>::Zero();
        Matrix<3, model::nv_size> jacr_dot = Matrix<3, model::nv_size>::Zero();
        mj_jac(mj_model_, mj_data_, jacp.data(), jacr.data(), points_.row(i).data(), body_ids_[i]);
        mj_jacDot(mj_model_, mj_data_, jacp_dot.data(), jacr_dot.data(), points_.row(i).data(), body_ids_[i]);
        int row_offset = i * 3;
        for(int row_idx = 0; row_idx < 3; row_idx++) {
            for(int col_idx = 0; col_idx < model::nv_size; col_idx++) {
                jacobian_translation(row_idx + row_offset, col_idx) = jacp(row_idx, col_idx);
                jacobian_rotation(row_idx + row_offset, col_idx) = jacr(row_idx, col_idx);
                jacobian_dot_translation(row_idx + row_offset, col_idx) = jacp_dot(row_idx, col_idx);
                jacobian_dot_rotation(row_idx + row_offset, col_idx) = jacr_dot(row_idx, col_idx);
            }
        }
    }
    
    Matrix<optimization::s_size, model::nv_size> taskspace_jacobian = Matrix<optimization::s_size, model::nv_size>::Zero();
    Matrix<optimization::s_size, model::nv_size> jacobian_dot = Matrix<optimization::s_size, model::nv_size>::Zero();
    taskspace_jacobian << jacobian_translation, jacobian_rotation;
    jacobian_dot << jacobian_dot_translation, jacobian_dot_rotation;
    Vector<optimization::s_size> taskspace_bias = Vector<optimization::s_size>::Zero();
    taskspace_bias = jacobian_dot * generalized_velocities;
    Matrix<model::nv_size, optimization::z_size> contact_jacobian = Matrix<model::nv_size, optimization::z_size>::Zero();
    contact_jacobian = jacobian_translation(Eigen::seq(Eigen::placeholders::end - Eigen::fix<optimization::z_size>, Eigen::placeholders::last), Eigen::placeholders::all).transpose();

    osc_data_.mass_matrix = mass_matrix;
    osc_data_.coriolis_matrix = coriolis_matrix;
    osc_data_.contact_jacobian = contact_jacobian;
    osc_data_.taskspace_jacobian = taskspace_jacobian;
    osc_data_.taskspace_bias = taskspace_bias;
    osc_data_.previous_q = generalized_positions;
    osc_data_.previous_qd = generalized_velocities;
}

// ===============================================================================================================
void OSCNode::update_optimization_data() {
    auto mass_matrix = matrix_utils::transformMatrix<double, model::nv_size, model::nv_size, matrix_utils::ColumnMajor>(osc_data_.mass_matrix.data());
    auto coriolis_matrix = matrix_utils::transformMatrix<double, model::nv_size, 1, matrix_utils::ColumnMajor>(osc_data_.coriolis_matrix.data());
    auto contact_jacobian = matrix_utils::transformMatrix<double, model::nv_size, optimization::z_size, matrix_utils::ColumnMajor>(osc_data_.contact_jacobian.data());
    auto taskspace_jacobian = matrix_utils::transformMatrix<double, optimization::s_size, model::nv_size, matrix_utils::ColumnMajor>(osc_data_.taskspace_jacobian.data());
    auto taskspace_bias = matrix_utils::transformMatrix<double, optimization::s_size, 1, matrix_utils::ColumnMajor>(osc_data_.taskspace_bias.data());
    auto desired_taskspace_ddx = matrix_utils::transformMatrix<double, model::site_ids_size, 6, matrix_utils::ColumnMajor>(taskspace_targets_.data());
    
    auto Aeq_matrix = evaluate_function<AeqParams>(Aeq_ops, {design_vector_.data(), mass_matrix.data(), coriolis_matrix.data(), contact_jacobian.data()});
    auto beq_matrix = evaluate_function<beqParams>(beq_ops, {design_vector_.data(), mass_matrix.data(), coriolis_matrix.data(), contact_jacobian.data()});
    auto Aineq_matrix = evaluate_function<AineqParams>(Aineq_ops, {design_vector_.data()});
    auto bineq_matrix = evaluate_function<bineqParams>(bineq_ops, {design_vector_.data()});
    auto H_matrix = evaluate_function<HParams>(H_ops, {design_vector_.data(), desired_taskspace_ddx.data(), taskspace_jacobian.data(), taskspace_bias.data()});
    auto f_matrix = evaluate_function<fParams>(f_ops, {design_vector_.data(), desired_taskspace_ddx.data(), taskspace_jacobian.data(), taskspace_bias.data()});

    opt_data_.H = H_matrix;
    opt_data_.f = f_matrix;
    opt_data_.Aeq = Aeq_matrix;
    opt_data_.beq = beq_matrix;
    opt_data_.Aineq = Aineq_matrix;
    opt_data_.bineq = bineq_matrix;
}

// ===============================================================================================================
absl::Status OSCNode::set_up_optimization(const Vector<model::contact_site_ids_size>& contact_mask) {
    MatrixColMajor<optimization::constraint_matrix_rows, optimization::constraint_matrix_cols> A;
    A << opt_data_.Aeq, opt_data_.Aineq, Abox_;
    Vector<optimization::bounds_size> lb;
    Vector<optimization::bounds_size> ub;
    Vector<optimization::z_size> z_lb_masked = z_lb_;
    Vector<optimization::z_size> z_ub_masked = z_ub_;
    
    for(int i = 0; i < model::contact_site_ids_size; i++) {
        z_lb_masked(Eigen::seqN(3 * i, 3)) *= contact_mask(i);
        z_ub_masked(Eigen::seqN(3 * i, 3)) *= contact_mask(i);
    }

    lb << opt_data_.beq, bineq_lb_, dv_lb_, u_lb_, z_lb_masked;
    ub << opt_data_.beq, opt_data_.bineq, dv_ub_, u_ub_, z_ub_masked;
    
    Eigen::SparseMatrix<double> sparse_H = opt_data_.H.sparseView();
    Eigen::SparseMatrix<double> sparse_A = A.sparseView();
    sparse_H.makeCompressed();
    sparse_A.makeCompressed();

    instance_.objective_matrix = sparse_H;
    instance_.objective_vector = opt_data_.f;
    instance_.constraint_matrix = sparse_A;
    instance_.lower_bounds = lb;
    instance_.upper_bounds = ub;
    
    absl::Status result = solver_.Init(instance_, settings_);
    return result;
}

// ===============================================================================================================
absl::Status OSCNode::update_optimization(const Vector<model::contact_site_ids_size>& contact_mask) {
    MatrixColMajor<optimization::constraint_matrix_rows, optimization::constraint_matrix_cols> A;
    A << opt_data_.Aeq, opt_data_.Aineq, Abox_;
    Vector<optimization::bounds_size> lb;
    Vector<optimization::bounds_size> ub;
    Vector<optimization::z_size> z_lb_masked = z_lb_;
    Vector<optimization::z_size> z_ub_masked = z_ub_;

    for(int i = 0; i < model::contact_site_ids_size; i++) {
        z_lb_masked(Eigen::seqN(3 * i, 3)) *= contact_mask(i);
        z_ub_masked(Eigen::seqN(3 * i, 3)) *= contact_mask(i);
    }
    
    lb << opt_data_.beq, bineq_lb_, dv_lb_, u_lb_, z_lb_masked;
    ub << opt_data_.beq, opt_data_.bineq, dv_ub_, u_ub_, z_ub_masked;
    
    Eigen::SparseMatrix<double> sparse_H = opt_data_.H.sparseView();
    Eigen::SparseMatrix<double> sparse_A = A.sparseView();
    sparse_H.makeCompressed();
    sparse_A.makeCompressed();


    absl::Status result;
    auto sparsity_check = solver_.UpdateObjectiveAndConstraintMatrices(sparse_H, sparse_A);
    if(sparsity_check.ok()) {
        result.Update(solver_.SetObjectiveVector(opt_data_.f));
        result.Update(solver_.SetBounds(lb, ub));
    } else {
        instance_.objective_matrix = sparse_H;
        instance_.objective_vector = opt_data_.f;
        instance_.constraint_matrix = sparse_A;
        instance_.lower_bounds = lb;
        instance_.upper_bounds = ub;
        result.Update(solver_.Init(instance_, settings_));
        result.Update(solver_.SetWarmStart(solution_, dual_solution_));
    }

    return result;
}

// ===============================================================================================================
bool OSCNode::solve_optimization() {
    exit_code_ = solver_.Solve();
    
    if (exit_code_ == OsqpExitCode::kOptimal) {
        solution_ = solver_.primal_solution();
        dual_solution_ = solver_.dual_solution();
        return true;
    } else {
        // Clear the solution so we don't accidentally use old values
        solution_.setZero(); 
        
        RCLCPP_WARN(this->get_logger(), "OSQP Solve Failed. Exit Code: %d", static_cast<int>(exit_code_));
        return false;
    }
}

// ===============================================================================================================
void OSCNode::reset_optimization() {
    Vector<optimization::constraint_matrix_cols> primal_vector = Vector<optimization::constraint_matrix_cols>::Zero();
    Vector<optimization::constraint_matrix_rows> dual_vector = Vector<optimization::constraint_matrix_rows>::Zero();
    std::ignore = solver_.SetWarmStart(primal_vector, dual_vector);
}

// ===============================================================================================================

void OSCNode::publish_torque_command(bool safety_override_active_local, 
                                     std::chrono::time_point<std::chrono::high_resolution_clock> state_read_time_local) 
{
    // --- Constants ---
    const std::set<std::string> reversed_joints_ = {
        "rear_left_hip", "rear_left_knee", "front_left_hip", "front_left_knee"};
    const std::array<std::string, model::nu_size> MOTOR_NAMES = {
        "rear_left_hip", "rear_left_knee", "rear_right_hip", "rear_right_knee",
        "front_left_hip", "front_left_knee", "front_right_hip", "front_right_knee"};
    
    const double MAX_TORQUE = 10.0;
    const int TORQUE_CONTROL_MODE = 1; 
    const int VELOCITY_CONTROL_MODE = 2; 

    // --- 1. Initialize Command Message ---
    auto command_msg = std::make_unique<Command>(); 
    command_msg->master_gain = 1.0; 
    command_msg->motor_commands.resize(model::nu_size);

    // --- 2. Determine Overall Mode and Populate Commands ---
    if (safety_override_active_local) {
        // SCENARIO A: PERMANENT SAFETY OVERRIDE
        command_msg->high_level_control_mode = 2; 
        
        for (size_t i = 0; i < model::nu_size; ++i) {
            command_msg->motor_commands[i].name = MOTOR_NAMES[i];
            
            // NOTE: HIGH GAIN HOLD IS THE RECOMMENDED SAFETY FIX (using Velocity mode is weak)
            command_msg->motor_commands[i].control_mode = VELOCITY_CONTROL_MODE;
            command_msg->motor_commands[i].position_setpoint = 0.0; 
            command_msg->motor_commands[i].velocity_setpoint = 0.0;
            command_msg->motor_commands[i].feedforward_torque = 0.0; 
            command_msg->motor_commands[i].kp = 0.0; 
            command_msg->motor_commands[i].kd = 0.0;
            command_msg->motor_commands[i].input_mode = 1;   
            command_msg->motor_commands[i].enable = true; 
        }

    } else {
        // SCENARIO B: NORMAL OPERATION 
        command_msg->high_level_control_mode = 2;
        
        Vector<model::nu_size> osc_torque = solution_(Eigen::seqN(optimization::dv_idx, optimization::u_size));

        std::stringstream ss;
        ss << "OSC Torques: [ ";
        for (size_t i = 0; i < model::nu_size; ++i) {
            ss << std::fixed << std::setprecision(2) << osc_torque(i) << " ";
        }
        ss << "]";
        RCLCPP_INFO(this->get_logger(), "%s", ss.str().c_str());
        

        for (size_t i = 0; i < model::nu_size; ++i) {
            double final_torque = osc_torque(i);
            
            if (reversed_joints_.count(MOTOR_NAMES[i])) {
                final_torque *= -1.0;
            }
            final_torque = std::clamp(final_torque, -MAX_TORQUE, MAX_TORQUE);

            command_msg->motor_commands[i].name = MOTOR_NAMES[i];
            command_msg->motor_commands[i].control_mode = TORQUE_CONTROL_MODE;
            command_msg->motor_commands[i].feedforward_torque = static_cast<double>(final_torque); 
            // Zero out unused PD terms
            command_msg->motor_commands[i].position_setpoint = 0.0;
            command_msg->motor_commands[i].velocity_setpoint = 0.0; 
            command_msg->motor_commands[i].kp = 0.0; 
            command_msg->motor_commands[i].kd = 0.0;
            command_msg->motor_commands[i].input_mode = 1;   
            command_msg->motor_commands[i].enable = true; 
        }
    }
    
    // --- Time Logging ---
    std::chrono::high_resolution_clock::time_point torque_ready_time_local = std::chrono::high_resolution_clock::now();
    
    // Calculate the control loop latency
    auto latency = std::chrono::duration_cast<std::chrono::microseconds>(
        torque_ready_time_local - state_read_time_local
    );
    double latency_ms = static_cast<double>(latency.count()) / 1000.0;

    // --- 3. Publish ---
    torque_publisher_->publish(std::move(command_msg));

    if (!safety_override_active_local) {
        RCLCPP_INFO(this->get_logger(), 
            "Latency: %.3f ms | SolvCode: %d |**OS Wait: %.3f ms** | Kinematics: %.3f ms | CasADi: %.3f ms | OSQP Solve: %.3f ms | Total Internal: %.3f ms",
            latency_ms,
            static_cast<int>(exit_code_),
            time_wait_for_execution_ms_, // NEW: Shows delay before computation starts            
            time_mujoco_update_ms_, 
            time_casadi_update_ms_, 
            time_osqp_solve_ms_,
            time_mujoco_update_ms_ + time_casadi_update_ms_ + time_osqp_solve_ms_);
    } else {
        RCLCPP_WARN(this->get_logger(), "Safety Override Active. Latency: %.3f ms", latency_ms);
    }
}


void OSCNode::stop_robot() {

    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        safety_override_active_ = true;        
    }    
    
    // 1. Constants for Safety Braking
    const double SAFETY_KP = 0.0; // High stiffness to hold position
    const double SAFETY_KD = 0.0;  // High damping to kill velocity
    const int POSITION_CONTROL_MODE = 3; // Ensure this matches your driver's Enum
    const int VELOCITY_CONTROL_MODE = 2; 

    // 2. Prepare Command
    auto command_msg = std::make_unique<Command>(); 
    command_msg->master_gain = 1.0; 
    command_msg->motor_commands.resize(model::nu_size);
    command_msg->high_level_control_mode = 2; // Safety / Idle mode

    const std::array<std::string, model::nu_size> MOTOR_NAMES = {
        "rear_left_hip", "rear_left_knee", "rear_right_hip", "rear_right_knee",
        "front_left_hip", "front_left_knee", "front_right_hip", "front_right_knee"};

    // 4. Fill Command: "Freeze at current position"
    for (size_t i = 0; i < model::nu_size; ++i) {
        command_msg->motor_commands[i].name = MOTOR_NAMES[i];
        
        // Switch to Position Mode (or Velocity Mode with 0 target)
        command_msg->motor_commands[i].control_mode = VELOCITY_CONTROL_MODE; 
        
        // Target = Last known position (Freeze)
        command_msg->motor_commands[i].position_setpoint = 0.0;
        command_msg->motor_commands[i].velocity_setpoint = 0.0;
        command_msg->motor_commands[i].feedforward_torque = 0.0; // CRITICAL: Zero torque
        
        // Set gains high to resist movement
        command_msg->motor_commands[i].kp = SAFETY_KP; 
        command_msg->motor_commands[i].kd = SAFETY_KD;
        command_msg->motor_commands[i].input_mode = 1;   
        command_msg->motor_commands[i].enable = true; 
    }

    // 5. Publish Immediate Stop
    // We assume the publisher is still valid because on_shutdown runs before destruction
    if (torque_publisher_) {
        torque_publisher_->publish(std::move(command_msg));
        RCLCPP_INFO(this->get_logger(), ">>> SAFETY STOP COMMAND SENT <<<");
        
        // Optional: Sleep briefly to ensure message hits the network before process dies
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}