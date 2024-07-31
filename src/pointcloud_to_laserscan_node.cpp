
#include "pointcloud_to_laserscan/pointcloud_to_laserscan_node.hpp"

#include <chrono>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/conversions.h>
// #include <pcl/filters/voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl_conversions/pcl_conversions.h>


#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "tf2_sensor_msgs/tf2_sensor_msgs.hpp"
#include "tf2_ros/create_timer_ros.h"

namespace pointcloud_to_laserscan
{

PointCloudToLaserScanNode::PointCloudToLaserScanNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("pointcloud_to_laserscan", options)
{
  target_frame_     = this->declare_parameter("target_frame", "");
  tolerance_        = this->declare_parameter("transform_tolerance", 0.01);
  input_queue_size_ = this->declare_parameter("queue_size", static_cast<int>(std::thread::hardware_concurrency()));
  min_height_       = this->declare_parameter("min_height", std::numeric_limits<double>::min());
  max_height_       = this->declare_parameter("max_height", std::numeric_limits<double>::max());
  angle_min_        = this->declare_parameter("angle_min", -M_PI);
  angle_max_        = this->declare_parameter("angle_max", M_PI);
  angle_increment_  = this->declare_parameter("angle_increment", M_PI / 180.0);
  scan_time_        = this->declare_parameter("scan_time", 1.0 / 30.0);
  range_min_        = this->declare_parameter("range_min", 0.0);
  range_max_        = this->declare_parameter("range_max", std::numeric_limits<double>::max());
  inf_epsilon_      = this->declare_parameter("inf_epsilon", 1.0);
  use_inf_          = this->declare_parameter("use_inf", true);
  active_output_    = this->declare_parameter("active_output", true);

  // Create service name with target frame as namespace
  std::string service_name = std::string(this->get_namespace()) +  std::string(this->get_name()) + std::string("/toggle_active_output");

  // Define the service server
  service_ = this->create_service<std_srvs::srv::SetBool>(
    service_name,
    std::bind(&PointCloudToLaserScanNode::toggleActiveOutput, this, std::placeholders::_1, std::placeholders::_2));

  
  pub_ = this->create_publisher<sensor_msgs::msg::LaserScan>("scan", rclcpp::SensorDataQoS());

  using std::placeholders::_1;
  // if pointcloud target frame specified, we need to filter by transform availability
  if (!target_frame_.empty()) {
    tf2_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(
      this->get_node_base_interface(), this->get_node_timers_interface());
    tf2_->setCreateTimerInterface(timer_interface);
    tf2_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf2_);
    message_filter_ = std::make_unique<MessageFilter>(
      sub_, *tf2_, target_frame_, input_queue_size_,
      this->get_node_logging_interface(),
      this->get_node_clock_interface());
    message_filter_->registerCallback(
      std::bind(&PointCloudToLaserScanNode::cloudCallback, this, _1));
  } else {  // otherwise setup direct subscription
    sub_.registerCallback(std::bind(&PointCloudToLaserScanNode::cloudCallback, this, _1));
  }

  subscription_listener_thread_ = std::thread(
    std::bind(&PointCloudToLaserScanNode::subscriptionListenerThreadLoop, this));
}

PointCloudToLaserScanNode::~PointCloudToLaserScanNode()
{
  alive_.store(false);
  subscription_listener_thread_.join();
}

void PointCloudToLaserScanNode::toggleActiveOutput(
  const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
  std::shared_ptr<std_srvs::srv::SetBool::Response> response)
{
  active_output_ = request->data;
  this->set_parameter(rclcpp::Parameter("active_output", active_output_));
  
  response->success = true;
  response->message = "Active output set to " + std::to_string(active_output_);

  RCLCPP_INFO(
    this->get_logger(),
    "Active output set to : %s",
    ((active_output_)?"true":"false"));
}

void PointCloudToLaserScanNode::subscriptionListenerThreadLoop()
{
  rclcpp::Context::SharedPtr context = this->get_node_base_interface()->get_context();

  const std::chrono::milliseconds timeout(100);
  while (rclcpp::ok(context) && alive_.load()) {
    int subscription_count = pub_->get_subscription_count() +
      pub_->get_intra_process_subscription_count();
    if (subscription_count > 0) {
      if (!sub_.getSubscriber()) {
        RCLCPP_INFO(
          this->get_logger(),
          "Got a subscriber to laserscan, starting pointcloud subscriber");
        rclcpp::SensorDataQoS qos;
        qos.keep_last(input_queue_size_);
        sub_.subscribe(this, "cloud_in", qos.get_rmw_qos_profile());
      }
    } else if (sub_.getSubscriber()) {
      RCLCPP_INFO(
        this->get_logger(),
        "No subscribers to laserscan, shutting down pointcloud subscriber");
      sub_.unsubscribe();
    }
    rclcpp::Event::SharedPtr event = this->get_graph_event();
    this->wait_for_graph_change(event, timeout);
  }
  sub_.unsubscribe();
}

void PointCloudToLaserScanNode::filterLaserScan(std::unique_ptr<sensor_msgs::msg::LaserScan> &scan_msg, double distance_threshold) {
  std::vector<float> filtered_ranges = scan_msg->ranges;
  size_t n = scan_msg->ranges.size();

  for (size_t i = 2; i < n - 2; ++i) {
    if (std::isfinite(scan_msg->ranges[i])) {
      double range = scan_msg->ranges[i];

      double avg_neighbour_range = (scan_msg->ranges[i - 2] + scan_msg->ranges[i - 1] +
                                    scan_msg->ranges[i + 1] + scan_msg->ranges[i + 2]) / 4.0;

      if (fabs(range - avg_neighbour_range) > distance_threshold) {
        filtered_ranges[i] = std::numeric_limits<float>::infinity();
      }
    }
  }

  scan_msg->ranges = filtered_ranges;
}

void PointCloudToLaserScanNode::cloudCallback(
  sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud_msg)
{
  // Convert ROS2 PointCloud2 to PCL PointCloud<pcl::PointXYZ>
  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::fromROSMsg(*cloud_msg, *pcl_cloud);

  // // Convert ROS2 PointCloud2 to PCL PointCloud<pcl::PointXYZ>
  // pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>());
  // pcl::fromROSMsg(*cloud_msg, *pcl_cloud);

  // // Apply Radius Outlier Removal filter
  // pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());
  // pcl::RadiusOutlierRemoval<pcl::PointXYZ> ror;
  // ror.setInputCloud(pcl_cloud);
  // ror.setRadiusSearch(0.05);  // Radius to search around each point
  // ror.setMinNeighborsInRadius(20);  // Minimum number of neighbors a point must have to remain in the cloud
  // ror.filter(*filtered_cloud);

  // // Convert filtered PCL PointCloud<pcl::PointXYZ> back to ROS2 PointCloud2
  // sensor_msgs::msg::PointCloud2 filtered_cloud_msg;
  // pcl::toROSMsg(*filtered_cloud, filtered_cloud_msg);
  // filtered_cloud_msg.header = cloud_msg->header;

  // // Use the filtered point cloud for further processing
  // cloud_msg = std::make_shared<sensor_msgs::msg::PointCloud2>(filtered_cloud_msg);


  // build laserscan output
  auto scan_msg = std::make_unique<sensor_msgs::msg::LaserScan>();
  scan_msg->header = cloud_msg->header;
  if (!target_frame_.empty()) {
    scan_msg->header.frame_id = target_frame_;
  }

  scan_msg->angle_min = angle_min_;
  scan_msg->angle_max = angle_max_;
  scan_msg->angle_increment = angle_increment_;
  scan_msg->time_increment = 0.0;
  scan_msg->scan_time = scan_time_;
  scan_msg->range_min = range_min_;
  scan_msg->range_max = range_max_;

  // determine amount of rays to create
  uint32_t ranges_size = std::ceil((scan_msg->angle_max - scan_msg->angle_min) / scan_msg->angle_increment);

  // determine if laserscan rays with no obstacle data will evaluate to infinity or max_range
  if (use_inf_) {
    scan_msg->ranges.assign(ranges_size, std::numeric_limits<double>::infinity());
  } else {
    scan_msg->ranges.assign(ranges_size, scan_msg->range_max + inf_epsilon_);
  }

  if(active_output_ == false){
    scan_msg->ranges.assign(ranges_size, std::numeric_limits<double>::infinity());
  
  } else {
  // Transform cloud if necessary
    if (scan_msg->header.frame_id != cloud_msg->header.frame_id) {
      try {
        auto cloud = std::make_shared<sensor_msgs::msg::PointCloud2>();
        tf2_->transform(*cloud_msg, *cloud, target_frame_, tf2::durationFromSec(tolerance_));
        cloud_msg = cloud;
      } catch (tf2::TransformException & ex) {
        RCLCPP_ERROR_STREAM(this->get_logger(), "Transform failure: " << ex.what());
        return;
      }
    }

    // Iterate through pointcloud
    for (sensor_msgs::PointCloud2ConstIterator<float> iter_x(*cloud_msg, "x"),
      iter_y(*cloud_msg, "y"), iter_z(*cloud_msg, "z");
      iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z)
    {
      if (std::isnan(*iter_x) || std::isnan(*iter_y) || std::isnan(*iter_z)) {
        RCLCPP_DEBUG(
          this->get_logger(),
          "rejected for nan in point(%f, %f, %f)\n",
          *iter_x, *iter_y, *iter_z);
        continue;
      }

      if (*iter_z > max_height_ || *iter_z < min_height_) {
        RCLCPP_DEBUG(
          this->get_logger(),
          "rejected for height %f not in range (%f, %f)\n",
          *iter_z, min_height_, max_height_);
        continue;
      }

      double range = hypot(*iter_x, *iter_y);
      if (range < range_min_) {
        RCLCPP_DEBUG(
          this->get_logger(),
          "rejected for range %f below minimum value %f. Point: (%f, %f, %f)",
          range, range_min_, *iter_x, *iter_y, *iter_z);
        continue;
      }
      if (range > range_max_) {
        RCLCPP_DEBUG(
          this->get_logger(),
          "rejected for range %f above maximum value %f. Point: (%f, %f, %f)",
          range, range_max_, *iter_x, *iter_y, *iter_z);
        continue;
      }

      double angle = atan2(*iter_y, *iter_x);
      if (angle < scan_msg->angle_min || angle > scan_msg->angle_max) {
        RCLCPP_DEBUG(
          this->get_logger(),
          "rejected for angle %f not in range (%f, %f)\n",
          angle, scan_msg->angle_min, scan_msg->angle_max);
        continue;
      }

      // overwrite range at laserscan ray if new range is smaller
      int index = (angle - scan_msg->angle_min) / scan_msg->angle_increment;
      if (range < scan_msg->ranges[index]) {
        scan_msg->ranges[index] = range;
      }
    }
  }
  
  // Apply the filtering
  filterLaserScan(scan_msg, 0.1);  // Use an appropriate distance threshold


  pub_->publish(std::move(scan_msg));
}

}  // namespace pointcloud_to_laserscan

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(pointcloud_to_laserscan::PointCloudToLaserScanNode)
