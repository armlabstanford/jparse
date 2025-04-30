#include <ros/ros.h>
#include <ros/package.h>
#include <sstream>  // for std::stringstream


#include <sensor_msgs/JointState.h>
#include <urdf/model.h>

#include <kdl_parser/kdl_parser.hpp>
#include <kdl/tree.hpp>
#include <kdl/chain.hpp>
#include <kdl/jntarray.hpp>
#include <kdl/chaindynparam.hpp>
#include <kdl/jntarrayvel.hpp>
#include <kdl/chainjnttojacsolver.hpp>

#include <Eigen/Dense>
#include <Eigen/SVD>

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Quaternion.h>

#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_broadcaster.h>

#include <manipulator_control/JparseTerms.h>
#include <manipulator_control/Matrow.h>
#include <manipulator_control/JParseSrv.h>


class JPARSE
{
public:
    JPARSE(ros::NodeHandle& nh);
    void Jnt_state_callback(const sensor_msgs::JointStateConstPtr& msg);
    void Jparse_calculation(const Eigen::MatrixXd& J,  Eigen::MatrixXd& J_parse, Eigen::MatrixXd& J_safety_nullspace, std::vector<int>& jparse_singular_index, Eigen::MatrixXd& U_safety, Eigen::VectorXd& S_new_safety, Eigen::MatrixXd& U_new_proj, Eigen::VectorXd& S_new_proj, Eigen::MatrixXd& U_new_sing, Eigen::VectorXd& Phi, double& gamma, double& singular_direction_gain_position, double& singular_direction_gain_angular);
    void Publish_JPARSE(const std_msgs::Header& header, const Eigen::MatrixXd& J_parse, const Eigen::MatrixXd& J_safety_nullspace);
    void JPARSE_visualization(const std_msgs::Header& header, const Eigen::MatrixXd& J_parse, const Eigen::MatrixXd& J_safety_nullspace, const std::vector<int>& jparse_singular_index, const Eigen::MatrixXd& U_safety, const Eigen::VectorXd& S_new_safety, const Eigen::MatrixXd& U_new_proj, const Eigen::VectorXd& S_new_proj, const Eigen::MatrixXd& U_new_sing, const Eigen::VectorXd& Phi);
    void matrix_to_msg(const Eigen::MatrixXd& mat, std::vector<manipulator_control::Matrow>& msg_rows);

    Eigen::MatrixXd pseudoInverse(const Eigen::MatrixXd& J, double tol = 1e-6);

    bool handleJparse(
        manipulator_control::JParseSrv::Request&  req,
        manipulator_control::JParseSrv::Response& resp
      );
    
private:
    ros::NodeHandle nh_;
    ros::Subscriber joint_state_sub_;
    ros::Publisher jparse_pub_;
    ros::Publisher jparse_markers_pub_;
    std::string root_, tip_;

    urdf::Model model_;
    KDL::Tree kdl_tree_;
    KDL::Chain kdl_chain_;
    KDL::Chain kdl_chain_service_; //for service option
    boost::shared_ptr<KDL::ChainDynParam> kdl_chain_dynamics_;
    boost::shared_ptr<KDL::ChainJntToJacSolver> jac_solver_;
    boost::shared_ptr<KDL::ChainJntToJacSolver> jac_solver_service_; //for service option
    
    KDL::JntArrayVel positions_;
    std::vector<std::string> joint_names_;

    tf2_ros::Buffer tfBuffer;
    std::unique_ptr<tf2_ros::TransformListener> tfListener;

    double gamma_; //Jparse threshold gamma
    double singular_direction_gain_position_;
    double singular_direction_gain_angular_; 

    //for service option
    bool            have_last_msg_ = false;
    std::mutex      last_msg_mutex_;
    sensor_msgs::JointStateConstPtr joint_state_msg_service_;
    ros::ServiceServer jparse_service_;

};

JPARSE::JPARSE(ros::NodeHandle& nh) : nh_(nh)
{

    ros::NodeHandle pnh("~");
    
    bool run_as_service = false;
    pnh.param("run_as_service", run_as_service, false);

    // Always subscribe, so we cache last_J_
    joint_state_sub_ = nh_.subscribe("joint_states", 1,
                                    &JPARSE::Jnt_state_callback, this);

    if (run_as_service)
    {
        // advertise the service
        jparse_service_ = nh_.advertiseService("jparse_srv",
                                            &JPARSE::handleJparse, this);
        ROS_INFO("JPARSE: running as service 'jparse_srv'");
    }

    pnh.param<std::string>("base_link_name", root_, "base_link");
    pnh.param<std::string>("end_link_name",   tip_,  "end_effector_link");

    nh_.param<double>("jparse_gamma", gamma_, 0.2);
    nh_.param<double>("singular_direction_gain_position", singular_direction_gain_position_, 1.0);
    nh_.param<double>("singular_direction_gain_angular", singular_direction_gain_angular_, 1.0);

    if (!model_.initParam("/robot_description") ||
    !kdl_parser::treeFromUrdfModel(model_, kdl_tree_))
    {
    ROS_ERROR("Failed to load /robot_description or build KDL tree");
    return;
    }

    //for getting the end-effector pose
    tfBuffer.setUsingDedicatedThread(true);
    tfListener.reset(new tf2_ros::TransformListener(tfBuffer));

    if (!kdl_tree_.getChain(root_, tip_, kdl_chain_))
    {
        ROS_ERROR("Failed to extract KDL chain from %s to %s", root_.c_str(), tip_.c_str());
        return;
    }

    kdl_chain_dynamics_.reset(new KDL::ChainDynParam(kdl_chain_, KDL::Vector(0, 0, -9.81)));
    jac_solver_.reset(new KDL::ChainJntToJacSolver(kdl_chain_));
    positions_ = KDL::JntArrayVel(kdl_chain_.getNrOfJoints());

    KDL::SetToZero(positions_.q);
    KDL::SetToZero(positions_.qdot);

    for (size_t i = 0; i < kdl_chain_.getNrOfSegments(); ++i)
    {
        KDL::Joint joint = kdl_chain_.getSegment(i).getJoint();
        if (joint.getType() != KDL::Joint::None)
            joint_names_.push_back(joint.getName());
    }

    joint_state_sub_ = nh_.subscribe("joint_states", 1, &JPARSE::Jnt_state_callback, this);
    jparse_pub_ = nh_.advertise<manipulator_control::JparseTerms>("jparse_output", 1);
    jparse_markers_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/jparse_ellipsoid_marker_cpp", 1);
}

void JPARSE::Jnt_state_callback(const sensor_msgs::JointStateConstPtr& msg)
{
    std::vector<double> q, dq;
    for (const auto& joint_name : joint_names_)
    {
        auto it = std::find(msg->name.begin(), msg->name.end(), joint_name);
        if (it != msg->name.end())
        {
            size_t idx = std::distance(msg->name.begin(), it);
            q.push_back(msg->position[idx]);
            dq.push_back(msg->velocity[idx]);
        }
    }

    if (q.size() != joint_names_.size()) return;

    for (size_t i = 0; i < joint_names_.size(); ++i)
    {
        positions_.q(i) = q[i];
        positions_.qdot(i) = dq[i];
    }

    KDL::Jacobian J_kdl(joint_names_.size());
    jac_solver_->JntToJac(positions_.q, J_kdl);
    Eigen::MatrixXd J = J_kdl.data;

    //store a copy for the service
    {
        std::lock_guard<std::mutex> guard(last_msg_mutex_);
        joint_state_msg_service_ = msg;
        have_last_msg_ = true;
    }

    Eigen::MatrixXd J_parse, J_safety_nullspace;
    //handle variable size kinematic chain
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(J, Eigen::ComputeFullU | Eigen::ComputeFullV);
    int n = svd.singularValues().size();
    std::vector<int> jparse_singular_index(n, 0); // Elements in this list are 0 if non-singular, 1 if singular
    Eigen::MatrixXd U_safety, U_new_proj, U_new_sing;
    Eigen::VectorXd S_new_safety, S_new_proj, Phi;

    Jparse_calculation(J, J_parse, J_safety_nullspace, jparse_singular_index, U_safety, S_new_safety, U_new_proj, S_new_proj, U_new_sing, Phi, gamma_, singular_direction_gain_position_, singular_direction_gain_angular_);
    Publish_JPARSE(msg->header, J_parse, J_safety_nullspace);
    
    Eigen::MatrixXd J_position = J.block(0, 0, 3, J.cols()); //extract J_v 
    Eigen::MatrixXd J_parse_position, J_safety_nullspace_position;

    //handle variable size kinematic chain
    Eigen::JacobiSVD<Eigen::MatrixXd> svd_position(J_position, Eigen::ComputeFullU | Eigen::ComputeFullV);
    int n_position = svd_position.singularValues().size();
    std::vector<int> jparse_singular_index_position(n_position, 0); // Elements in this list are 0 if non-singular, 1 if singular
    Eigen::MatrixXd U_safety_position, U_new_proj_position, U_new_sing_position;
    Eigen::VectorXd S_new_safety_position, S_new_proj_position, Phi_position;

    Jparse_calculation(J_position, J_parse_position, J_safety_nullspace_position, jparse_singular_index_position, U_safety_position, S_new_safety_position, U_new_proj_position, S_new_proj_position, U_new_sing_position, Phi_position, gamma_, singular_direction_gain_position_, singular_direction_gain_angular_);    
    JPARSE_visualization(msg->header, J_parse_position, J_safety_nullspace_position, jparse_singular_index_position, U_safety_position, S_new_safety_position, U_new_proj_position, S_new_proj_position, U_new_sing_position, Phi_position); 
}

Eigen::MatrixXd JPARSE::pseudoInverse(const Eigen::MatrixXd& J, double tol)
{
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(J, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd& singularValues = svd.singularValues();
    Eigen::MatrixXd S_pinv = Eigen::MatrixXd::Zero(svd.matrixV().cols(), svd.matrixU().cols());

    for (int i = 0; i < singularValues.size(); ++i)
    {
        if (singularValues(i) > tol)
        {
            S_pinv(i, i) = 1.0 / singularValues(i);
        }
    }

    return svd.matrixV() * S_pinv * svd.matrixU().transpose();
}

void JPARSE::Jparse_calculation(const Eigen::MatrixXd& J,  Eigen::MatrixXd& J_parse, Eigen::MatrixXd& J_safety_nullspace, std::vector<int>& jparse_singular_index, Eigen::MatrixXd& U_safety, Eigen::VectorXd& S_new_safety, Eigen::MatrixXd& U_new_proj, Eigen::VectorXd& S_new_proj, Eigen::MatrixXd& U_new_sing, Eigen::VectorXd& Phi, double& gamma, double& singular_direction_gain_position, double& singular_direction_gain_angular)
{
    /*
    Steps are as follows:
    1. Find the SVD of J
    2. Find the adjusted condition number and Jparse singular index
    3. Find the Projection Jacobian
    4. Find the Safety Jacobian
    5. Find the singular direction projection components
    6. Find the pseudo inverse of J_safety and J_proj
    7. Find Jparse
    8. Find the null space of J_safety
    */

    //1. Find the SVD of J
    Eigen::MatrixXd U, V;
    Eigen::VectorXd S;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(J, Eigen::ComputeFullU | Eigen::ComputeFullV);
    U = svd.matrixU();
    S = svd.singularValues();
    V = svd.matrixV();

    if (U.rows() == U.cols() && U.determinant() < 0.0) {
        int k = U.cols() - 1;     // pick the last column
        U.col(k) *= -1.0;
        V.col(k) *= -1.0;
    }

    U_safety = U; //Safety Jacobian shares the same U as the SVD of J

    //2. find the adjusted condition number
    double max_singular_value = S.maxCoeff();
    std::vector<double> adjusted_condition_numbers(S.size());
    for (int i = 0; i < S.size(); ++i)
    {
        adjusted_condition_numbers[i] = S(i) / max_singular_value;
    }
    
    //3. Find the Projection Jacobian
    std::vector<int> valid_indices;

    for (int i = 0; i < S.size(); ++i)
    {
        // keep only the elements whose singular value is greater than the threshold
        if (S(i) > gamma * max_singular_value)
        {
            valid_indices.push_back(i);
        }else{
            jparse_singular_index[i] = 1; // set the index to 1 if the singular value is less than the threshold
        }
    }

    U_new_proj = Eigen::MatrixXd(U.rows(), valid_indices.size());
    S_new_proj = Eigen::VectorXd(valid_indices.size());

    for (size_t i = 0; i < valid_indices.size(); ++i)
    {
        U_new_proj.col(i) = U.col(valid_indices[i]);
        S_new_proj(i) = S(valid_indices[i]);
    }
    Eigen::MatrixXd S_new_proj_matrix = Eigen::MatrixXd::Zero(U_new_proj.cols(), V.rows());
    for (int i = 0; i < S_new_proj.size(); ++i)
    {
        S_new_proj_matrix(i, i) = S_new_proj(i);
    }
    Eigen::MatrixXd J_proj = U_new_proj * S_new_proj_matrix * V.transpose();

    //4. Find the Safety Jacobian
    S_new_safety = Eigen::VectorXd(S.size());
    for (int i = 0; i < S.size(); ++i)
    {
        //if the singular value is greater than the threshold, keep it otherwise set it to the threshold
        if ((S(i) / max_singular_value) > gamma)
        {
            S_new_safety(i) = S(i);
        }
        else
        {
            S_new_safety(i) = gamma * max_singular_value;
        }
    }


    Eigen::MatrixXd S_new_safety_matrix = Eigen::MatrixXd::Zero(U.rows(), V.cols());
    for (int i = 0; i < S_new_safety.size(); ++i)
    {
        S_new_safety_matrix(i, i) = S_new_safety(i);
    }
    Eigen::MatrixXd J_safety = U * S_new_safety_matrix * V.transpose();

    //5. Find the singular direction projection components
    Eigen::MatrixXd Phi_matrix, Kp_singular, Phi_singular;
    std::vector<int> valid_indices_sing;
    bool set_empty_bool = true; // set to true if the valid indices are empty
    for (int i = 0; i < S.size(); ++i)
    {
        if (adjusted_condition_numbers[i] <= gamma)
        {
            set_empty_bool = false;
            valid_indices_sing.push_back(i);
        }
    }

    U_new_sing = Eigen::MatrixXd(U.rows(), valid_indices_sing.size());
    Phi = Eigen::VectorXd(valid_indices_sing.size());

    if (set_empty_bool==false)
    {
        for (size_t i = 0; i < valid_indices_sing.size(); ++i)
        {
            U_new_sing.col(i) = U.col(valid_indices_sing[i]);
            Phi(i) = adjusted_condition_numbers[valid_indices_sing[i]] / gamma;
        }

        Phi_matrix = Eigen::MatrixXd::Zero(Phi.size(), Phi.size());
        for (int i = 0; i < Phi.size(); ++i)
        {
            Phi_matrix(i, i) = Phi(i);
        }
        
        Kp_singular = Eigen::MatrixXd::Zero(U.rows(), U.cols());
        for (int i = 0; i < 3; ++i)
        {
            Kp_singular(i, i) = singular_direction_gain_position;
        }
        if (Kp_singular.cols() > 3) //checks if position only (J_v) or full jacobian
        {
            for (int i = 3; i < 6; ++i)
            {
            Kp_singular(i, i) = singular_direction_gain_angular;
            }
        }
        Phi_singular = U_new_sing * Phi_matrix  * U_new_sing.transpose() * Kp_singular; // put it all together
    }

    //6. Find pseudo inverse of J_safety and J_proj
    Eigen::MatrixXd J_safety_pinv = pseudoInverse(J_safety);
    Eigen::MatrixXd J_proj_pinv = pseudoInverse(J_proj);

    //7. Find the Jparse
    if (set_empty_bool==false)
    {
        J_parse = J_safety_pinv * J_proj * J_proj_pinv + J_safety_pinv * Phi_singular;
    }
    else
    {
        J_parse =  J_safety_pinv * J_proj * J_proj_pinv;
    }

    //8. Find the null space of J_safety
    J_safety_nullspace = Eigen::MatrixXd::Identity(J_safety.cols(), J_safety.cols()) - J_safety_pinv * J_safety;

}

void JPARSE::JPARSE_visualization(const std_msgs::Header& header, const Eigen::MatrixXd& J_parse, const Eigen::MatrixXd& J_safety_nullspace, const std::vector<int>& jparse_singular_index, const Eigen::MatrixXd& U_safety, const Eigen::VectorXd& S_new_safety, const Eigen::MatrixXd& U_new_proj, const Eigen::VectorXd& S_new_proj, const Eigen::MatrixXd& U_new_sing, const Eigen::VectorXd& Phi)    
{
    //This script takes in the J_Parse matricies and visualizes them in RVIZ; this is done for position. 

    //Get the end-effector position and orientation
    geometry_msgs::TransformStamped transformStamped;
    try
    {
        transformStamped = tfBuffer.lookupTransform(root_, tip_, ros::Time(0));
    }
    catch (tf2::TransformException& ex)
    {
        ROS_WARN("%s", ex.what());
        return;
    }


    //1. create a marker array
    visualization_msgs::MarkerArray marker_array;
    visualization_msgs::Marker J_safety_marker;
    visualization_msgs::Marker J_proj_marker;
    visualization_msgs::Marker J_singular_marker;

    //2. Set up the J_safety_marker
    J_safety_marker.header = header;
    J_safety_marker.header.frame_id = root_;
    J_safety_marker.ns = "J_safety";
    J_safety_marker.id = 0;
    J_safety_marker.type = visualization_msgs::Marker::SPHERE;
    J_safety_marker.action = visualization_msgs::Marker::ADD;
    //set the marker pose position to the end effector position
    J_safety_marker.pose.position.x = transformStamped.transform.translation.x;
    J_safety_marker.pose.position.y = transformStamped.transform.translation.y;
    J_safety_marker.pose.position.z = transformStamped.transform.translation.z;
    double ellipsoid_scale = 0.25;
    
    double safety_value_1 = std::max(0.001, S_new_safety(0));
    J_safety_marker.scale.x = ellipsoid_scale * safety_value_1;

    double safety_value_2 = std::max(0.001, S_new_safety(1));
    J_safety_marker.scale.y = ellipsoid_scale * safety_value_2;

    double safety_value_3 = std::max(0.001, S_new_safety(2));
    J_safety_marker.scale.z = ellipsoid_scale * safety_value_3;

    Eigen::Matrix3d R = U_safety.block<3,3>(0, 0);  
    

    Eigen::Quaterniond q_safety(R);
    //normalize the quaternion
    q_safety.normalize();  // optional if R is already perfectly orthonormal
    J_safety_marker.pose.orientation.x = q_safety.x();
    J_safety_marker.pose.orientation.y = q_safety.y();
    J_safety_marker.pose.orientation.z = q_safety.z();
    J_safety_marker.pose.orientation.w = q_safety.w();

    J_safety_marker.color.a = 0.7;
    J_safety_marker.color.r = 1.0;
    J_safety_marker.color.g = 0.0;
    J_safety_marker.color.b = 0.0;

    // Add the J_safety_marker to the marker array
    marker_array.markers.push_back(J_safety_marker);


    // Determine if J_proj and J_singular exist
    bool J_proj_exists = false;
    bool J_singular_exists = false;
    int number_of_singular_directions = 0;
    for (int i = 0; i < 3; ++i) {
        if (jparse_singular_index[i] == 0) {
            // some directions are non-singular
            J_proj_exists = true;
        } else if (jparse_singular_index[i] == 1) {
            // some directions are singular
            J_singular_exists = true;
            number_of_singular_directions++;
        } 
    }


       

    //2. setup the J_proj_marker if it exists
    if(J_proj_exists==true){
        // Set up the J_proj_marker
        J_proj_marker.header = header;
        J_proj_marker.header.frame_id = root_;
        J_proj_marker.ns = "J_proj";
        J_proj_marker.id = 1;
        J_proj_marker.type = visualization_msgs::Marker::SPHERE;
        J_proj_marker.action = visualization_msgs::Marker::ADD;


        if(jparse_singular_index[0]==0){ 
            double proj_value_1 = std::max(0.001, S_new_proj(0));
            J_proj_marker.scale.x = ellipsoid_scale * proj_value_1;
        }else{
            J_proj_marker.scale.x = 0.001;
        }
    
        if(jparse_singular_index[1]==0){ 
            double proj_value_2 = std::max(0.001, S_new_proj(1));
            J_proj_marker.scale.y = ellipsoid_scale * proj_value_2;
        }else{
            J_proj_marker.scale.y = 0.001;
        }
    
        if(jparse_singular_index[2]==0){
            double proj_value_3 = std::max(0.001, S_new_proj(2));
            J_proj_marker.scale.z = ellipsoid_scale * proj_value_3;
        }else{
            J_proj_marker.scale.z = 0.001;
        }

        J_proj_marker.pose.position.x = transformStamped.transform.translation.x;
        J_proj_marker.pose.position.y = transformStamped.transform.translation.y;
        J_proj_marker.pose.position.z = transformStamped.transform.translation.z;

        Eigen::Matrix3d R_proj = U_safety.block<3,3>(0, 0);  
        Eigen::Quaterniond q_proj(R_proj);

        //normalize the quaternion
        q_proj.normalize();  // optional if R is already perfectly orthonormal
        J_proj_marker.pose.orientation.x = q_proj.x();
        J_proj_marker.pose.orientation.y = q_proj.y();
        J_proj_marker.pose.orientation.z = q_proj.z();
        J_proj_marker.pose.orientation.w = q_proj.w();

        J_proj_marker.color.a = 0.7;
        J_proj_marker.color.r = 0.0;
        J_proj_marker.color.g = 0.0;
        J_proj_marker.color.b = 1.0;
    }

    // Add the J_proj_marker to the marker array    
    if(J_proj_exists==true){
        marker_array.markers.push_back(J_proj_marker);
    }


    //3. setup the J_singular_marker if it exists

    if (J_singular_exists)
    {
        // Extract end-effector position once
        geometry_msgs::Point ee_pos;
        ee_pos.x = transformStamped.transform.translation.x;
        ee_pos.y = transformStamped.transform.translation.y;
        ee_pos.z = transformStamped.transform.translation.z;

        double arrow_scale = 1.0;

        for (int idx = 0; idx < number_of_singular_directions; ++idx)
        {
            visualization_msgs::Marker marker;
            // --- Header & identity ---
            marker.header = header;                            // copy stamp & frame_id
            marker.header.frame_id = root_;
            marker.ns     = "J_singular";
            marker.id     = idx + 2;
            marker.type   = visualization_msgs::Marker::ARROW;
            marker.action = visualization_msgs::Marker::ADD;
            marker.lifetime = ros::Duration(0.1);

            // --- Start point ---
            geometry_msgs::Point start_point = ee_pos;

            // --- Decide arrow direction ---
            Eigen::Vector3d u_col = U_new_sing.block<3,1>(0, idx);  // first 3 rows of column
            double dot = ee_pos.x * u_col.x()
                    + ee_pos.y * u_col.y()
                    + ee_pos.z * u_col.z();

            Eigen::Vector3d arrow_dir = (dot < 0.0 ? u_col : -u_col);

            // --- End point ---
            double mag = std::abs(Phi(idx));
            geometry_msgs::Point end_point;
            end_point.x = ee_pos.x + arrow_scale * arrow_dir.x() * mag;
            end_point.y = ee_pos.y + arrow_scale * arrow_dir.y() * mag;
            end_point.z = ee_pos.z + arrow_scale * arrow_dir.z() * mag;

            // --- Push points (clear just in case) ---
            marker.points.clear();
            marker.points.push_back(start_point);
            marker.points.push_back(end_point);

            // --- Fixed arrow sizing ---
            marker.scale.x = 0.01;  // shaft diameter
            marker.scale.y = 0.05;  // head diameter
            marker.scale.z = 0.05;  // head length

            // --- Color = solid red ---
            marker.color.r = 1.0;
            marker.color.g = 0.0;
            marker.color.b = 0.0;
            marker.color.a = 1.0;

            marker_array.markers.push_back(marker);
        }

    }
  
    //publish the marker array
    jparse_markers_pub_.publish(marker_array);
}

bool JPARSE::handleJparse(
    manipulator_control::JParseSrv::Request&  req,
    manipulator_control::JParseSrv::Response& res)
  {
    // find the jacobian for the joints specified in the request

    std::string root_service = req.base_link_name;
    std::string tip_service = req.end_link_name;
    std::vector<std::string> joint_names;
    double gamma_service = req.gamma;
    double singular_direction_gain_position_service = req.singular_direction_gain_position;
    double singular_direction_gain_angular_service = req.singular_direction_gain_angular;
    sensor_msgs::JointStateConstPtr msg;

    ROS_INFO("JPARSE service: Received request for base_link: %s, end_link: %s", root_service.c_str(), tip_service.c_str());
    //setup the KDL chain
    if (!kdl_tree_.getChain(root_service, tip_service, kdl_chain_service_))
    {
        ROS_ERROR("Failed to extract KDL chain from %s to %s", root_service.c_str(), tip_service.c_str());
        return false;
    }
    jac_solver_service_.reset(new KDL::ChainJntToJacSolver(kdl_chain_service_));

    KDL::JntArrayVel positions = KDL::JntArrayVel(kdl_chain_service_.getNrOfJoints());

    KDL::SetToZero(positions.q);
    KDL::SetToZero(positions.qdot);

    for (size_t i = 0; i < kdl_chain_service_.getNrOfSegments(); ++i)
    {
        KDL::Joint joint = kdl_chain_service_.getSegment(i).getJoint();
        if (joint.getType() != KDL::Joint::None)
        joint_names.push_back(joint.getName());
    }


    //send over the copy of the joint state message
    {
        std::lock_guard<std::mutex> guard(last_msg_mutex_);
        if (!have_last_msg_) {
            ROS_WARN("No joint state yet, refusing service");
            return false;
          }
        msg = joint_state_msg_service_;
        have_last_msg_ = false;
        joint_state_msg_service_.reset();
    }

    std::vector<double> q, dq;
    for (const auto& joint_name : joint_names)
    {
        auto it = std::find(msg->name.begin(), msg->name.end(), joint_name);
        if (it != msg->name.end())
        {
            size_t idx = std::distance(msg->name.begin(), it);
            q.push_back(msg->position[idx]);
            dq.push_back(msg->velocity[idx]);
        }
    }

    if (q.size() != joint_names.size())
    {
        ROS_ERROR("Joint state message does not contain all joint names");
        return false;
    } 

    for (size_t i = 0; i < joint_names.size(); ++i)
    {
        positions.q(i) = q[i];
        positions.qdot(i) = dq[i];
    }

    KDL::Jacobian J_kdl(joint_names.size());
    jac_solver_service_->JntToJac(positions.q, J_kdl);
    Eigen::MatrixXd J = J_kdl.data;
  
    // Now run usual pipeline on J
    Eigen::MatrixXd J_parse, J_safety_nullspace;

    //handle any kinematic chain size
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(J, Eigen::ComputeFullU | Eigen::ComputeFullV);
    int n = svd.singularValues().size();
    std::vector<int> jparse_singular_index(n, 0);// Elements in this list are 0 if non-singular, 1 if singular

    Eigen::MatrixXd U_safety, U_new_proj, U_new_sing;
    Eigen::VectorXd S_new_safety, S_new_proj, Phi;

    Jparse_calculation(J, J_parse, J_safety_nullspace, jparse_singular_index, U_safety, S_new_safety, U_new_proj, S_new_proj, U_new_sing, Phi, gamma_service, singular_direction_gain_position_service, singular_direction_gain_angular_service);

    // Fill response
    matrix_to_msg(J_parse,            res.jparse);
    matrix_to_msg(J_safety_nullspace, res.jsafety_nullspace);
    return true;
  }

void JPARSE::matrix_to_msg(const Eigen::MatrixXd& mat, std::vector<manipulator_control::Matrow>& msg_rows)
{
    msg_rows.clear();
    for (int i = 0; i < mat.rows(); ++i)
    {
        manipulator_control::Matrow row;
        for (int j = 0; j < mat.cols(); ++j)
        {
            row.row.push_back(mat(i, j));
        }
        msg_rows.push_back(row);
    }
}

void JPARSE::Publish_JPARSE(const std_msgs::Header& header, const Eigen::MatrixXd& J_parse, const Eigen::MatrixXd& J_safety_nullspace)
{
    manipulator_control::JparseTerms msg;
    msg.header = header;

    matrix_to_msg(J_parse, msg.jparse);
    matrix_to_msg(J_safety_nullspace, msg.jsafety_nullspace);

    jparse_pub_.publish(msg);
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "jparse_cpp_node");
    ros::NodeHandle nh;
    JPARSE parser(nh);
    
    ros::AsyncSpinner spinner(2); // AsyncSpinner with 2 threads lets your subscriber and service, callbacks run concurrently.
    spinner.start();
    ros::waitForShutdown();
    return 0;
}

