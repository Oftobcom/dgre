import grpc
from concurrent import futures
import time

# Import proto-generated files
# from services import PolicyRequest_pb2_grpc, PolicyResponse_pb2, RiskEvaluationServicer

# Placeholder: replace with your actual proto imports
class RiskEvaluationServicer:
    def EvaluatePolicy(self, request, context):
        # TODO: convert proto request -> Agent list
        # Call PolicyEngine.apply_policies
        # Convert back to PolicyResponse proto
        pass

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    # Add servicer here
    # PolicyRequest_pb2_grpc.add_RiskEvaluationServicer_to_server(RiskEvaluationServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("DG RE gRPC server started on port 50051")
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == "__main__":
    serve()
