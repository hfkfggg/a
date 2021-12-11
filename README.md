package four;
import java.util.Scanner;
public class m {
	public static void main(String[] args) {
		// TODO Auto-generated method stub
				String letterS;
				char letter;
				System.out.print("Enter a letter: ");
				Scanner input = new Scanner(System.in);
				letterS = input.next();
				if(letterS.length() != 1)
					System.exit(1);
				letter = letterS.charAt(0);
				if(Character.isLetter(letter))
				{	letter = Character.toUpperCase(letter);
					if("AEIOU".indexOf(letter) != -1)
						System.out.println(letterS + " is a vowel");
					else
						System.out.println(letterS + " is a consonant");}
				else
					System.out.println(letter + " is an invalid input");
				input.close();}
}
