package ExtractResults;

import java.util.Comparator;

public class ObjectiveRecords implements Comparator<ObjectiveRecords>{

	private double featureRate, error ;

	public ObjectiveRecords(double featureRate, double error){
		this.featureRate = featureRate;
		this.error = error;
	}

	//return 1 if me is better than other
	//return 0 if no one is better
	//return -1 if me is worse than other
	public int compareTo(ObjectiveRecords other) {
		if(this.featureRate < other.featureRate){
			if(this.error <= other.error)
				return 1;
			else
				return 0;
		}
		else if(this.featureRate > other.featureRate){
			if(this.error >= other.error)
				return -1;
			else return 0;
		}
		else{
			if(this.error < other.error)
				return 1;
			else if(this.error == other.error)
				return 0;
			else return -1;
		}
	}


	public boolean equals(Object other){
		if(!(other instanceof ObjectiveRecords))
			return false;
		ObjectiveRecords orOther = (ObjectiveRecords) other;
		if(orOther.error == this.error && orOther.featureRate == this.featureRate)
			return true;
		return false;
	}

	public int compare(ObjectiveRecords o1, ObjectiveRecords o2) {
		if (o2.featureRate < o1.featureRate)
			return -1;
		else if (o2.featureRate > o1.featureRate)
			return 1;
		else
			return 0;
	}

	public double getFeatureRate(){
		return this.featureRate;
	}

	public double getErrorRate(){
		return this.error;
	}

	public String toString(){
		return this.featureRate+","+this.error;
	}

}
