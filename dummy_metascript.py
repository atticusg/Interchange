import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("script", type=str)
    parser.add_argument("-o", "--output", type=str)
    args = parser.parse_args()
    print("Dummy metascript:")
    print(f"Output: {args.output}, Script: {args.script}")

