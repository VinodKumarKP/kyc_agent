import logging
from typing import Dict, Any

from fastmcp import FastMCP

# from pydantic import BaseModel # Removed unused import

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("name-lookup-tools")


CUSTOMER_DB = {
    "john_doe_1234": {
        "customer_id": "CUST_001",
        "full_name": "John Doe",
        "account_number": "****-****-****-1234",
        "full_account_number": "4532-1234-5678-1234",
        "credit_score": 750,
        "credit_limit": 5000.00,
        "credit_balance": 1250.00,
        "account_status": "active",
        "phone": "555-0123",
        "email": "john.doe@email.com"
    },
    "jane_smith_5678": {
        "customer_id": "CUST_002",
        "full_name": "Jane Smith",
        "account_number": "****-****-****-5678",
        "full_account_number": "4532-9876-5432-5678",
        "credit_score": 680,
        "credit_limit": 3000.00,
        "credit_balance": 890.50,
        "account_status": "active",
        "phone": "555-0456",
        "email": "jane.smith@email.com"
    }
}

TRANSACTION_DB = {
    "CUST_001": [
        {"date": "2024-01-25", "description": "Amazon Purchase", "amount": -125.99, "balance": 1250.00, "category": "shopping"},
        {"date": "2024-01-22", "description": "Payment Received", "amount": 500.00, "balance": 1375.99, "category": "payment"},
        {"date": "2024-01-20", "description": "Shell Gas Station", "amount": -45.50, "balance": 875.99, "category": "fuel"},
        {"date": "2024-01-18", "description": "Whole Foods", "amount": -89.34, "balance": 921.49, "category": "grocery"},
        {"date": "2024-01-15", "description": "Starbucks", "amount": -6.75, "balance": 1010.83, "category": "dining"}
    ],
    "CUST_002": [
        {"date": "2024-01-24", "description": "Target Purchase", "amount": -67.89, "balance": 890.50, "category": "shopping"},
        {"date": "2024-01-21", "description": "Payment Received", "amount": 300.00, "balance": 958.39, "category": "payment"},
        {"date": "2024-01-19", "description": "Uber Ride", "amount": -15.25, "balance": 658.39, "category": "transportation"},
        {"date": "2024-01-17", "description": "Netflix Subscription", "amount": -15.99, "balance": 673.64, "category": "entertainment"}
    ]
}

@mcp.tool()
def verify_customer(self, name: str, last_four_digits: str) -> Dict[str, Any]:
    """Verify customer identity"""
    lookup_key = f"{name.lower().replace(' ', '_')}_{last_four_digits}"

    if lookup_key in CUSTOMER_DB:
        customer_data = CUSTOMER_DB[lookup_key].copy()
        customer_data["verified"] = True
        return customer_data
    else:
        return {
            "verified": False,
            "error": "Customer not found or invalid credentials"
        }

@mcp.tool()
def get_account_balance(customer_id: str) -> Dict[str, Any]:
    """Get current account balance"""

    try:
        customer_data = None
        for key, data in CUSTOMER_DB.items():
            if data["customer_id"] == customer_id:
                customer_data = data
                break

        if not customer_data:
            return {"error": "Customer not found"}

        return {
            "customer_id": customer_id,
            "credit_limit": customer_data["credit_limit"],
            "current_balance": customer_data["credit_balance"],
            "available_credit": customer_data["credit_limit"] - customer_data["credit_balance"],
            "utilization_percentage": (customer_data["credit_balance"] / customer_data["credit_limit"]) * 100
        }
    except Exception as ex:
        return {"error": "Error retrieving account balance" + str(ex)}

@mcp.tool()
def get_transaction_history(customer_id: str, limit: int = 10) -> Dict[str, Any]:
    """Get transaction history"""
    if customer_id not in TRANSACTION_DB:
        return {"error": "No transaction history found"}

    transactions = TRANSACTION_DB[customer_id][:limit]

    return {
        "customer_id": customer_id,
        "transaction_count": len(transactions),
        "transactions": transactions
    }

@mcp.tool()
def get_credit_score(customer_id: str) -> Dict[str, Any]:
    """Get credit score"""
    customer_data = None
    for key, data in CUSTOMER_DB.items():
        if data["customer_id"] == customer_id:
            customer_data = data
            break

    if not customer_data:
        return {"error": "Customer not found"}

    score = customer_data["credit_score"]

    if score >= 800:
        rating = "Excellent"
    elif score >= 740:
        rating = "Very Good"
    elif score >= 670:
        rating = "Good"
    elif score >= 580:
        rating = "Fair"
    else:
        rating = "Poor"

    return {
        "customer_id": customer_id,
        "credit_score": score,
        "credit_rating": rating,
        "score_range": "300-850",
        "factors": [
            "Payment history: 35% of score",
            "Credit utilization: 30% of score",
            "Length of credit history: 15% of score",
            "Credit mix: 10% of score",
            "New credit: 10% of score"
        ],
        "last_updated": "2024-01-28"
    }

@mcp.tool()
def request_credit_limit_increase(customer_id: str, requested_amount: float, reason: str = "") -> Dict[
    str, Any]:
    """Request credit limit increase"""
    customer_data = None
    for key, data in CUSTOMER_DB.items():
        if data["customer_id"] == customer_id:
            customer_data = data
            break

    if not customer_data:
        return {"error": "Customer not found"}

    current_limit = customer_data["credit_limit"]

    if requested_amount >= 7000:
        return  {"error": "Requested amount exceeds maximum allowable limit"}

    if requested_amount <= current_limit:
        return {"error": "Requested amount must be higher than current limit"}

    if requested_amount > current_limit * 3:
        return {"error": "Requested amount exceeds maximum allowable increase"}

    request_id = f"CLR_{hash(f'{customer_id}_{requested_amount}') % 100000:05d}"

    credit_score = customer_data["credit_score"]
    utilization = (customer_data["credit_balance"] / current_limit) * 100

    if credit_score >= 750 and utilization < 30 and requested_amount <= current_limit * 1.5:
        status = "approved"
        processing_days = 1
    elif credit_score >= 700 and utilization < 50:
        status = "under_review"
        processing_days = 3
    else:
        status = "pending_manual_review"
        processing_days = 7

    return {
        "request_id": request_id,
        "customer_id": customer_id,
        "current_limit": current_limit,
        "requested_limit": requested_amount,
        "status": status,
        "estimated_processing_days": processing_days,
        "reason": reason,
        "submitted_date": "2024-01-28"
    }

@mcp.tool()
def get_account_summary(customer_id: str) -> Dict[str, Any]:
    """Get account summary"""
    customer_data = None
    for key, data in CUSTOMER_DB.items():
        if data["customer_id"] == customer_id:
            customer_data = data
            break

    if not customer_data:
        return {"error": "Customer not found"}

    recent_transactions = TRANSACTION_DB.get(customer_id, [])[:5]

    return {
        "customer_info": {
            "customer_id": customer_data["customer_id"],
            "name": customer_data["full_name"],
            "account_number": customer_data["account_number"],
            "account_status": customer_data["account_status"]
        },
        "credit_info": {
            "credit_limit": customer_data["credit_limit"],
            "current_balance": customer_data["credit_balance"],
            "available_credit": customer_data["credit_limit"] - customer_data["credit_balance"],
            "credit_score": customer_data["credit_score"]
        },
        "recent_transactions": recent_transactions
    }

if __name__ == "__main__":
    mcp.run()
